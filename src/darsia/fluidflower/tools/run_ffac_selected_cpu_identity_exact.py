"""New isolated exact-convention identity stages; no Policy A dispatch."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib
import json
import multiprocessing
from pathlib import Path
from unittest.mock import patch

from darsia.fluidflower.phase_ffac_selected_cpu_identity_exact import ExactIdentityEvaluator
from darsia.fluidflower.phase_ffac_selected_phase import asset, bound_json
from darsia.fluidflower.tools.refit_ffac_g3_downstream import atomic_json, canonical_sha256, utc_now
from darsia.fluidflower.tools import run_ffac_selected_cpu_identity as original

MODULES = original.MODULES + ('darsia.fluidflower.phase_ffac_selected_cpu_identity_exact',
    'darsia.fluidflower.tools.run_ffac_selected_cpu_identity_exact')


def bind(source_plan, identity_inventory, root):
    root = Path(root)
    baseline = original.bind(source_plan, identity_inventory, root/'base_binding')
    plan = bound_json(baseline)
    binding = dict(plan['phase_campaign_binding'])
    binding['code'] = {name: {'sha256': asset(importlib.import_module(name).__file__)['sha256']}
                       for name in MODULES}
    binding.update(unchanged_corrected_operator=False,
        spacing_or_previous_valid_conventions_prefixed=True,
        exact_conventions={'spacing': 'exact matched saved original y/x values',
            'modes': ['spacing_only', 'both'],
            'both_previous_native_alignment_mask': None,
            'decoder_and_feature_support_masks': 'unchanged',
            'extended_previous_alignment_mask': 'unchanged'},
        parent_identity_base=baseline)
    plan.update(phase_campaign_binding=binding, pipeline_sha256=canonical_sha256(binding))
    path = root/'identity_plan.json'
    if path.exists(): raise FileExistsError('Exact identity plan already exists')
    atomic_json(path, plan)
    return asset(path)


def sequence(plan, root, run, last, mode):
    def factory(*positional, **named):
        return ExactIdentityEvaluator(*positional, mode=mode, **named)
    with patch.object(original, 'IdentityEvaluator', factory):
        return original.run_sequence(plan, root, run, last)


def run_stage(plan, root, stage, workers):
    root = Path(root)
    if stage == 'spacing-only': runs, mode = [('ac19', 1)], 'spacing_only'
    elif stage == 'both-ac19': runs, mode = [('ac19', 1)], 'both'
    elif stage == 'pilots':
        if bound_json(asset(root/'both-ac19'/'stage.json'))['complete'] is not True:
            raise ValueError('Both-convention AC19 sequence must pass first')
        runs, mode = [('ac19', 1), ('ac18', 5)], 'both'
    else:
        if bound_json(asset(root/'pilots'/'stage.json'))['complete'] is not True:
            raise ValueError('Both complete pilot histories must pass before all39')
        runs, mode = [(run, 1) for run in sorted(bound_json(plan)['runs'])], 'both'
    stage_root = root/stage
    if stage_root.exists(): raise FileExistsError('Stage output already exists; no silent retry')
    stage_root.mkdir(parents=True)
    results = []
    if stage != 'all39' or workers == 1:
        for run, last in runs:
            result = sequence(plan, stage_root, run, last, mode); results.append(result)
            if result['complete'] is not True: break
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            pending = [pool.submit(sequence, plan, str(stage_root), run, last, mode) for run, last in runs]
            for future in as_completed(pending):
                result = future.result(); results.append(result)
                if result['complete'] is not True:
                    for task in pending: task.cancel()
                    break
    complete = len(results) == len(runs) and all(row['complete'] for row in results)
    record = {'contract': original.CONTRACT, 'stage': stage, 'created_utc': utc_now(),
        'identity_plan': plan, 'mode': mode, 'complete': complete,
        'all_39_complete': complete and stage == 'all39', 'cpu_only': True,
        'labels_bit_identical': complete, 'stored_probabilities_bit_identical': complete,
        'results': results, 'policy_a_production_allowed': False,
        'training': False, 'mass_computed': False}
    atomic_json(stage_root/'stage.json', record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['bind','spacing-only','both-ac19','pilots','all39'], required=True)
    parser.add_argument('--root', required=True); parser.add_argument('--plan')
    parser.add_argument('--source-plan'); parser.add_argument('--identity-inventory')
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args(argv)
    if args.stage == 'bind': print(json.dumps(bind(args.source_plan, args.identity_inventory, args.root))); return 0
    if not 1 <= args.workers <= 24: raise ValueError('CPU workers must be bounded 1..24')
    record = run_stage(asset(args.plan), args.root, args.stage, args.workers)
    print(json.dumps(record), flush=True)
    return 0 if record['complete'] else 2


if __name__ == '__main__': raise SystemExit(main())
