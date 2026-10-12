"""Prepare only the measured RAW references declared for a common campaign.

This invokes the source-bound V5 processor; it does not run phase inference,
train an optical model, fit mass parameters, or read a held-out phase frame.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import traceback


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _initialize():
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    import cv2
    cv2.setNumThreads(1)


def _prepare(row):
    started = time.perf_counter()
    from darsia.fluidflower.tools.ffac_v5_normalized_clearwater import prepare_one
    from darsia.fluidflower.tools.ffac_track1_v4r5_source_processor import Track1V4R5SourceProcessor
    import numpy as np
    args = dict(row['prepare_one_args'])
    args = {key: value if key == 'run' else Path(value) for key, value in args.items()}
    output = Path(args['output'])
    supplemental = output / 'common_reference_transport.json'
    if output.exists():
        raise FileExistsError(f'Existing reference folder is immutable: {output}')
    # These are only frozen CO2-free geometric reference metadata.
    native = json.loads(args['native_manifest_path'].read_text(encoding='utf-8-sig'))
    aggregate = json.loads(args['photo_aggregate_path'].read_text(encoding='utf-8-sig'))
    source_sha = native['v5_photometry']['source_manifest_sha256']
    matching = [item for item in aggregate['rows'] if item['run'] == args['run'] and item['source_manifest_sha256'] == source_sha]
    if len(matching) != 1:
        raise ValueError('One frozen CO2-free B geometry is required')
    processor = Track1V4R5SourceProcessor(matching[0], calibration_manifest_path=args['calibration_manifest_path'], photo_aggregate_path=args['photo_aggregate_path'])
    base = processor.cohort.frame(0)
    matrix = np.linalg.inv(np.asarray(base['H_v5_B_to_frame_native'], np.float64)) @ np.asarray(base['H_ac28_to_frame_native'], np.float64)
    del processor
    result = prepare_one(**args)
    transport = dict(contract='ffac-common-native-clearwater-transport-v1', run=args['run'],
        clearwater_receipt=result['path'], clearwater_receipt_sha256=result['sha256'],
        H_ac28_to_clearwater_grid=matrix.tolist(), baseline_grid='native-B-2979x5472',
        current_frame_transport='H_ac28_to_current_rig @ inv(H_ac28_to_clearwater_grid)',
        missing_support='unquantified; never extrapolated or fitted',
        created_utc=datetime.now(timezone.utc).isoformat(), no_model_fit=True,
        source_generator_sha256=_sha(__import__('inspect').getfile(prepare_one)))
    with supplemental.open('x', encoding='utf-8') as stream:
        json.dump(transport, stream, indent=2, allow_nan=False)
    return dict(run=args['run'], complete=True, reference=result,
        transport=dict(path=str(supplemental), sha256=_sha(supplemental)),
        wall_seconds=time.perf_counter()-started)


def run(task_path, output, *, workers=8, heldout=False, freeze=None):
    tasks = json.loads(Path(task_path).read_text(encoding='utf-8-sig'))
    rows = [row for row in tasks['rows'] if bool(row['execution_after_freeze']) == heldout]
    if heldout:
        from darsia.fluidflower.phase_ffac_common_mass_production import FrozenCalibrationEvidence
        FrozenCalibrationEvidence.verify(freeze)
    if len(rows) != (12 if heldout else 27) or len({row['run'] for row in rows}) != len(rows):
        raise ValueError('Exact fixed reference cohort required')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_initialize) as pool:
        futures = {pool.submit(_prepare, row): row for row in rows}
        for future in as_completed(futures):
            row = futures[future]
            try:
                value = future.result()
            except Exception as exc:
                value = dict(run=row['run'], complete=False, error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
            with (output / (row['run']+'.json')).open('x', encoding='utf-8') as stream:
                json.dump(value, stream, indent=2, allow_nan=False)
            results.append(value)
            print(json.dumps({key: value[key] for key in ('run','complete','error','wall_seconds') if key in value}), flush=True)
    report = dict(contract='ffac-common-native-reference-preparation-v1',
        heldout=heldout, frozen_calibration=str(freeze) if freeze else None,
        task_manifest_sha256=_sha(task_path), complete=all(row['complete'] for row in results),
        workers=workers, wall_seconds=time.perf_counter()-started,
        results=sorted(results, key=lambda row: row['run']), model_fit=False, phase_inference=False)
    with (output / 'result.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    if not report['complete']:
        raise RuntimeError('One or more clearwater references failed; preserve failures and diagnose before retry')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--heldout', action='store_true')
    parser.add_argument('--freeze')
    args = parser.parse_args()
    run(args.tasks,args.output,workers=args.workers,heldout=args.heldout,freeze=args.freeze)


if __name__ == '__main__':
    main()
