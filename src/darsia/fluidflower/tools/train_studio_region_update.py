"""Freeze current Studio reviews, fit regional heads and prepare a new review model."""
from __future__ import annotations

from darsia.fluidflower.tools._paths import project_path, source_path, process_directory, series_root, series_path, tool_path, local_cache_root, package_root

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import uuid

from darsia.fluidflower.tools import train_region_coverage_overnight as training
from darsia.fluidflower import phase_runtime
from darsia.fluidflower.tools.prepare_optical_region_studio import read, write, prepare
from darsia.fluidflower.project import sha256_file, utc_now


def release_training_accelerator():
    """Do not retain the training allocator while a child owns inference VRAM."""
    import gc
    gc.collect()
    if 'cupy' not in sys.modules:
        return {'backend': 'cpu'}
    import cupy as cp
    cp.cuda.get_current_stream().synchronize()
    pool = cp.get_default_memory_pool()
    before = pool.total_bytes()
    pool.free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    return dict(allocator_bytes_before=before, allocator_bytes_after=pool.total_bytes())


def prepare_review_extras(model_root, job_root, *, prepare_disagreements=False):
    """Only an explicit request builds a disagreement queue before review."""
    if prepare_disagreements:
        subprocess.run([sys.executable, '-u', '-m', 'darsia.fluidflower.tools.build_phase_disagreement_review',
                        '--model', str(model_root), '--output', str(Path(job_root) / 'disagreements.json')],
                       cwd=process_directory(), check=True)
    else:
        write(Path(job_root) / 'disagreements_deferred.json', dict(status='deferred',
            action='Check disagreements', model_root=str(model_root),
            reason='Calculated on request in ordinary Phase Model Studio'))


class TrainingTiming:
    """Durable wall-time evidence for one invocation, including interrupted runs."""

    def __init__(self, root, *, model_root, series, capture_only=False):
        self.root = Path(root)
        self.started_utc = utc_now()
        self.invocation_id = self.started_utc.replace(':', '').replace('+', '_') + '-' + uuid.uuid4().hex[:12]
        self.started_monotonic = time.perf_counter()
        self.stage_started = self.started_monotonic
        self.stage_started_utc = self.started_utc
        self.stage = 'initialization'
        self.stages = []
        self.metadata = dict(model_root=str(model_root), series=series,
                             capture_only=bool(capture_only), pid=os.getpid())
        self.event_path = self.root / 'timing_events' / (self.invocation_id + '.jsonl')
        self.event_path.parent.mkdir(parents=True, exist_ok=True)
        self._record('started', self.started_monotonic, self.started_utc)

    def _record(self, event, now, utc, *, outcome='running', error=None):
        elapsed = now - self.started_monotonic
        value = dict(format='phase-training-timing-v1', invocation_id=self.invocation_id,
                     event=event, utc=utc, elapsed_seconds=elapsed,
                     monotonic_seconds=now, stage=self.stage, outcome=outcome)
        if error is not None:
            value['error'] = error
        with self.event_path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(value, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        summary = dict(format='phase-training-timing-v1', invocation_id=self.invocation_id,
                       started_utc=self.started_utc, updated_utc=utc,
                       elapsed_seconds=elapsed, outcome=outcome, **self.metadata,
                       completed_stages=self.stages, event_path=str(self.event_path),
                       active_stage=(dict(stage=self.stage, started_utc=self.stage_started_utc,
                           elapsed_seconds=now-self.stage_started) if outcome == 'running' else None))
        if error is not None:
            summary['error'] = error
        if outcome != 'running':
            summary['finished_utc'] = utc
        write(self.root / 'timing_runs' / (self.invocation_id + '.json'), summary)
        write(self.root / 'timing.json', summary)

    def transition(self, stage):
        if stage == self.stage:
            return
        now, utc = time.perf_counter(), utc_now()
        self._close_stage(now, utc)
        self.stage, self.stage_started, self.stage_started_utc = stage, now, utc
        self._record('stage_started', now, utc)

    def _close_stage(self, now, utc):
        self.stages.append(dict(stage=self.stage, started_utc=self.stage_started_utc,
                                finished_utc=utc, duration_seconds=now-self.stage_started,
                                start_elapsed_seconds=self.stage_started-self.started_monotonic,
                                end_elapsed_seconds=now-self.started_monotonic))

    def finish(self, outcome, *, error=None):
        now, utc = time.perf_counter(), utc_now()
        self._close_stage(now, utc)
        self._record('finished', now, utc, outcome=outcome, error=error)


def run(args):
    timing = TrainingTiming(args.job.resolve(), model_root=args.output.resolve(),
                            series=args.series, capture_only=args.capture_only)
    try:
        result = _run(args, timing)
    except BaseException as exc:
        timing.finish('interrupted' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'failed',
                      error=traceback.format_exc())
        raise
    timing.finish('capture_only_ready' if args.capture_only else 'ready_for_review')
    return result


def aqueous_annotation_receipt(evidence, *, required_review=None, existing_receipt=None):
    """Keep the dense-aqueous audit without excluding gas-only/quiz updates."""
    if existing_receipt:
        wanted = set(existing_receipt['required_ids'])
    elif required_review is not None:
        required = read(required_review)
        wanted = {a['id'] for row in required['decisions'].values()
                  for a in row.get('annotations', [])
                  if a.get('category') == 'missed_visible_aqueous_co2'}
    else:
        wanted = {r['payload']['id'] for r in evidence['regions']
                  if r['payload'].get('category') == 'missed_visible_aqueous_co2'}
    selected = [r for r in evidence['regions'] if r['payload'].get('id') in wanted]
    present = {r['payload']['id'] for r in selected}
    if ((required_review is not None and not wanted) or present != wanted
            or any(r['phase_class'] != 'aqueous' for r in selected)):
        raise ValueError(f'New dense aqueous annotations missing or misclassified: {sorted(wanted-present)}')
    return existing_receipt or dict(required_review=str(required_review) if required_review else None,
        required_review_sha256=sha256_file(required_review) if required_review else None,
        selection_scope='explicit-review' if required_review else 'all-effective-missed-aqueous',
        required_ids=sorted(wanted), captured_ids=sorted(present),
        new_region_ids=[r['region_id'] for r in selected],
        counts=dict(Counter(r['run'] for r in selected)), total_regions=len(evidence['regions']),
        phase_classes=dict(Counter(r['phase_class'] for r in evidence['regions'])))


def _run(args, timing):
    root = args.job.resolve()
    root.mkdir(parents=True, exist_ok=True)
    def status(stage, **extra):
        timed_stage = {
            'capturing_reviews': 'capture',
            'extracting_training_samples': 'extraction',
            'fitting_regional_heads': 'fit',
            'trained': 'model_preparation',
            'preparing_studio': 'phase_maps_and_studio',
            'studio_ready': 'studio_publication',
        }.get(stage)
        if timed_stage:
            timing.transition(timed_stage)
        value = dict(stage=stage, updated_utc=utc_now(), model_root=str(args.output), **extra)
        write(root / 'status.json', value)
        print(stage, flush=True)
        callback = getattr(args, 'status_callback', None)
        if callback is not None:
            callback(value)
    def command(module, *options):
        subprocess.run([sys.executable, '-u', '-m', module, *map(str, options)],
                       cwd=process_directory(), check=True)
    config = dict(series=args.series, parent=str(args.parent.resolve()),
        collection=str(args.parent.resolve() / 'review_source'), root=str(root),
        model_root=str(args.output.resolve()), cache=str(args.cache.resolve()))
    actor = str(getattr(args, 'actor', '') or '').strip()
    if actor:
        config['created_by'] = actor
    from darsia.fluidflower.phase_execution_resources import available_resources
    available_cpus, available_memory = available_resources()
    cpu_selection = None
    if args.cpus is not None:
        cpus = args.cpus
    elif 'PHASE_CPU_LIMIT' in os.environ:
        cpus = int(os.environ['PHASE_CPU_LIMIT'])
    elif args.backend != 'cpu':
        from darsia.fluidflower.tools.phase_throughput_resources import total_cpu_recommendation
        cpu_selection = total_cpu_recommendation(available_cpus,
            args.output.resolve().parent.parent / 'config' / 'phase_execution_calibration.json',
            generate_review_proposals=False)
        cpus = cpu_selection['cpu_limit']
    else:
        cpus = available_cpus
    memory_gb = args.memory_gb if args.memory_gb is not None else available_memory / 1024**3
    if cpus < 1 or memory_gb <= 0:
        raise ValueError('CPU and memory budgets must be positive')
    config['cpu_threads'] = cpus
    config['feature_cache_root'] = str(args.feature_cache_root or args.cache.parent.parent / 'regional_image_features' / args.series)
    write(root / 'execution_resources.json', dict(cpus=cpus, memory_gb=memory_gb,
                                                cpu_selection=cpu_selection))
    try:
        status('capturing_reviews')
        evidence = training.capture(config)
        if getattr(args, 'photometry_revision_root', None) is not None:
            from darsia.fluidflower.tools.phase_training_photometry_revision import (
                bind_captured_evidence, verify_revision)
            verified = verify_revision(args.photometry_revision_root)
            shutil.copy2(root / 'evidence.json', root / 'captured_evidence.json')
            evidence, revised_collection = bind_captured_evidence(
                evidence, Path(config['collection']), root / 'photometry_source_binding', verified)
            config['collection'] = str(revised_collection)
            write(root / 'evidence.json', evidence)
            write(root / 'photometry_source_selection.json', dict(
                format='phase-training-photometry-revision-v1',
                original_collection=str(args.parent / 'review_source'),
                revised_collection=str(revised_collection),
                capture_snapshot=str(root / 'captured_evidence.json'),
                revision_receipt=str(verified['receipt_path']),
                revision_receipt_sha256=verified['receipt_sha256'],
                source_manifest_sha256=verified['source_manifest_sha256'],
                target_manifest_sha256=verified['target_manifest_sha256'],
                changed_frame_indices=verified['changed_frame_indices']))
        timing.transition('preflight')
        # Verify individual annotation IDs, not just a count or session title.
        receipt_path = root / 'new_annotation_receipt.json'
        existing_receipt = read(receipt_path) if receipt_path.is_file() else None
        receipt = aqueous_annotation_receipt(evidence, required_review=args.required_review,
                                             existing_receipt=existing_receipt)
        present = set(receipt['captured_ids'])
        write(root / 'new_annotation_receipt.json', receipt)
        plan, bindings = training.bind_domains(evidence, Path(config['collection']))
        backend = args.backend
        if backend == 'auto':
            try:
                import cupy as cp
                if cp.cuda.runtime.getDeviceCount() < 1:
                    raise RuntimeError('No CUDA device')
                max_pixels = max(int(g['native']['analysis_image_shape_hw'][0])*int(g['native']['analysis_image_shape_hw'][1]) for g in plan)
                if cp.cuda.runtime.memGetInfo()[0] < max_pixels*260 + 512*1024**2:
                    raise RuntimeError('Insufficient free GPU memory for full-resolution optical features')
                backend = 'cuda'
            except (ImportError, OSError, RuntimeError):
                backend = 'cpu'
        config['backend'] = backend
        write(root / 'domain_bindings.json', bindings)
        status('preflight_ready', new_aqueous_regions=len(present), total_regions=len(evidence['regions']), frames=len(plan))
        if args.capture_only:
            return
        snapshot = root / 'code_snapshot'
        snapshot.mkdir(exist_ok=True)
        for source in (Path(__file__), Path(training.__file__), Path(training.extraction.__file__),
                       Path(training.regional.__file__), Path(training.joint_aqueous.__file__),
                       project_path('scripts/phase_region_feature_cache.py'),
                       project_path('scripts/prepare_optical_region_studio.py'),
                       project_path('scripts/phase_reference_batch.py'),
                       project_path('scripts/phase_reference_readiness.py'),
                       project_path('scripts/finish_optical_region_studio.py'),
                       project_path('external/darsia/src/darsia/fluidflower/phase_execution_resources.py'),
                       project_path('external/darsia/src/darsia/fluidflower/phase_regional_gas.py'),
                       project_path('external/darsia/src/darsia/fluidflower/phase_free_gas_output_guard.py'),
                       project_path('external/darsia/src/darsia/fluidflower/phase_bottom_gas_guard.py'),
                       project_path('external/darsia/src/darsia/fluidflower/phase_region_edge_guard.py')):
            shutil.copy2(source, snapshot / source.name)
        with training.threadpool_limits(limits=cpus):
            status('extracting_training_samples', new_aqueous_regions=len(present))
            extracted = training.extract(config, evidence, plan, bindings)
            new_ids = set(receipt['new_region_ids'])
            coverage = [r for r in extracted['coverage'] if r['region_id'] in new_ids]
            if {r['region_id'] for r in coverage} != new_ids:
                raise ValueError('Extraction did not account for every new annotation')
            write(root / 'new_annotation_training_coverage.json', dict(regions=coverage,
                contributing=sum(r['training_rows'] > 0 for r in coverage), total=len(new_ids)))
            if args.required_contributing_regions is not None:
                required_regions=set(read(args.required_contributing_regions)['required_ids'])
                contributing={r['region_id'] for r in extracted['coverage'] if r['training_rows']>0}
                if required_regions-contributing:
                    raise ValueError(f'Required preserved regions have no training rows: {sorted(required_regions-contributing)}')
                write(root/'preserved_region_coverage.json',dict(required_ids=sorted(required_regions),
                    contributing_ids=sorted(required_regions), policy='native-points-with-interpolated-regional-features'))
            status('fitting_regional_heads', new_aqueous_regions=len(present))
            fitted = training.fit(config, evidence, extracted)
        status('trained', experiment=str(fitted))
        write(root / 'training_accelerator_release.json', release_training_accelerator()
              if backend == 'cuda' else dict(backend='cpu'))
        prepare(fitted, Path(config['collection']), args.output, fresh_optical=True,
                bottom_gas_guard=bool(getattr(args, 'bottom_gas_guard', False)),
                region_edge_guard=bool(getattr(args, 'region_edge_guard', False)),
                metric_postprocessing=bool(getattr(args, 'metric_postprocessing', True)),
                nominal_review_gas_minimum=bool(getattr(
                    args, 'nominal_review_gas_minimum', False)))
        # Track-1 preview is a detached Studio candidate.  Put the requested
        # review runs first before any map worker reads its own manifest;
        # source manifests, annotation coordinates, and the parent stay intact.
        if (phase_runtime.load_physics_profile(args.output) or {}).get(
                'v5_track1_review_scope') is not None:
            review_manifest_path = args.output / 'review_source' / 'postflight_manifest.json'
            review_manifest = read(review_manifest_path)
            priority = {'ac28': 0, 'ac45': 1, 'ac56': 2, 'ac22': 3}
            original_jobs = list(review_manifest['jobs'])
            review_manifest['jobs'] = sorted(
                original_jobs,
                key=lambda job: priority.get(str(job['run']).casefold(), 4))
            write(review_manifest_path, review_manifest)
            write(args.output / 'studio_review_priority.json', dict(
                contract='ffac-v5-track1-first-maps-priority-v1',
                requested_runs=['ac28', 'ac45', 'ac56', 'ac22'],
                ordered_runs=[job['run'] for job in review_manifest['jobs']],
                detached_candidate_only=True))
        write(args.output / 'expected_review_runs.json', dict(runs=[j['run'] for j in read(Path(config['collection'])/'postflight_manifest.json')['jobs']]))
        request = read(args.output / 'training_request.json')
        if actor:
            request['created_by'] = actor
        request['settings']['iteration'] = (
            'FF_AC V5 RAW · Photo-only supported sand · NOT FOR MASS ANALYSIS · Studio review'
            if bool(getattr(args, 'nominal_review_gas_minimum', False))
            and (phase_runtime.load_physics_profile(args.output) or {}).get(
                'v5_track1_review_scope') is not None else
            'Reviewed optical regions + learned gas + nominal 2.5 mm2 gas minimum; Studio review only'
            if bool(getattr(args, 'nominal_review_gas_minimum', False)) else
            'Reviewed aqueous regions + learned gas + 2.5 mm2 gas minimum'
            if bool(getattr(args, 'metric_postprocessing', True)) else
            'Reviewed optical regions + learned gas; metric postprocessing disabled')
        write(args.output / 'training_request.json', request)
        status('preparing_studio')
        preparation_options = []
        if getattr(args, 'verify_numerics', False):
            preparation_options.append('--verify-numerics')
        if getattr(args, 'eager_references', False):
            preparation_options.append('--eager-references')
        command('darsia.fluidflower.tools.prepare_optical_region_studio', '--root', args.output, '--resume',
                '--fresh-optical', '--workers', '1', '--cpus', cpus, '--memory-gb', memory_gb,
                '--backend', args.backend, *preparation_options)
        if not read(args.output / 'studio_preparation.json').get('presentation_ready'):
            timing.transition('studio_finalization')
            command('darsia.fluidflower.tools.finish_optical_region_studio', '--root', args.output, '--cpus', cpus, '--memory-gb', memory_gb)
        status('studio_ready')
        if getattr(args, 'prepare_disagreements', False):
            timing.transition('disagreements')
        prepare_review_extras(args.output, root,
            prepare_disagreements=getattr(args, 'prepare_disagreements', False))
        status('ready_for_review', disagreements_deferred=not getattr(args, 'prepare_disagreements', False))
    except Exception:
        status('failed', error=traceback.format_exc())
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('parent', 'output', 'job', 'cache'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--required-review', type=Path)
    parser.add_argument('--required-contributing-regions', type=Path)
    parser.add_argument('--series', choices=('ffum','ffac','ffbm'), required=True)
    parser.add_argument('--capture-only', action='store_true')
    parser.add_argument('--cpus', type=int)
    parser.add_argument('--memory-gb', type=float)
    parser.add_argument('--feature-cache-root', type=Path)
    parser.add_argument('--backend', choices=('auto','cpu','cuda'), default='auto')
    parser.add_argument('--verify-numerics', action='store_true', help='Run explicit CPU numerical diagnostics.')
    parser.add_argument('--eager-references', action='store_true', help='Prepare reference phase maps before opening Studio.')
    parser.add_argument('--prepare-disagreements', action='store_true', help='Build disagreement list now instead of on request.')
    run(parser.parse_args(argv))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
