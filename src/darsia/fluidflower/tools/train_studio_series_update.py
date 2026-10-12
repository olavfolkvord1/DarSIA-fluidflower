"""Studio launcher for the common AC/UM/BM regional training workflow.

The selected model supplies frozen lineage and its registered review timeline.
The existing updater captures all eligible saved corrections, blind labels and
quiz answers; this adapter neither selects individual bundles nor changes fits.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from types import SimpleNamespace
import uuid

from darsia.fluidflower import phase_runtime, phase_review_retractions
from darsia.fluidflower.phase_training.scope import series_id
from darsia.fluidflower.review_storage import resolve_review_path
from darsia.fluidflower.tools._paths import local_cache_root, series_path
from darsia.fluidflower.tools import train_studio_region_update as update


EVENT_PREFIX = 'PHASE_CONTINUATION_EVENT '


def record_photometry_revision(args):
    """Expose the verified source revision before native map materialization."""
    if args.photometry_revision_root is None:
        return
    from darsia.fluidflower.tools.phase_training_photometry_revision import verify_revision
    verified = verify_revision(args.photometry_revision_root)
    native = update.read(verified['root'] / 'native_frames/ac47/manifest.json')
    detail = dict(format='phase-training-photometry-revision-v1',
        photometry_revision_id=native['photometry_revision_id'],
        algorithm_revision='neutral-tone-only-v4', run='ac47', dataset='ffac',
        receipt_path=str(verified['receipt_path']),
        receipt_sha256=verified['receipt_sha256'],
        source_manifest_sha256=verified['source_manifest_sha256'],
        target_manifest_sha256=verified['target_manifest_sha256'],
        changed_frame_indices=verified['changed_frame_indices'],
        concentration_mass_calibration=dict(
            status='revalidation_required', compatible_with_parent=False,
            warning=('Existing AC concentration and mass calibration was fitted on the '
                     'parent photometry. Revalidate or recalibrate it against this '
                     'corrected source before using concentration or mass estimates.')))
    request_path = args.output / 'training_request.json'
    summary_path = args.output / 'summary.json'
    request, summary = update.read(request_path), update.read(summary_path)
    request['photometry_revision'] = detail
    request.setdefault('settings', {})['photometry_revision_id'] = detail['photometry_revision_id']
    summary['photometry_revision'] = detail
    summary['photometry_revision_id'] = detail['photometry_revision_id']
    summary['concentration_mass_calibration_ready'] = False
    update.write(request_path, request)
    update.write(summary_path, summary)


def prepare_launch(parent, library, cache, selected, actor, *, backend='auto',
                   cpus=None, memory_gb=None, feature_cache_root=None,
                   photometry_revision_root=None):
    """Validate scope and allocate new paths without capturing or fitting."""
    selected = series_id(selected)
    actor = str(actor).strip()
    if selected not in {'ffac', 'ffum', 'ffbm'} or not actor:
        raise ValueError('Select AC, UM or BM and enter a reviewer name before training')
    if backend not in {'auto', 'cpu', 'cuda'}:
        raise ValueError('Training backend must be auto, cpu or cuda')
    parent, library, cache = (resolve_review_path(Path(value)).resolve()
                              for value in (parent, library, cache))
    expected_library = resolve_review_path(series_path(selected, 'models')).resolve()
    if library != expected_library:
        raise ValueError('New models must be stored in the selected series model folder')
    if library.is_relative_to(parent) or cache.is_relative_to(parent):
        raise ValueError('New model and cache folders must be outside the selected parent model')
    summary = update.read(parent / 'summary.json')
    previous = update.read(parent / 'training_request.json')
    if (previous.get('evaluation_only') or previous.get('settings', {}).get('evaluation_only')
            or summary.get('artifact_kind') == 'phase-transfer-evaluation'):
        raise ValueError('Select a fitted source model; transfer validation requires explicit training inclusion')
    datasets = {series_id(value) for value in summary.get('inventory', {}).get('datasets', [])}
    if datasets != {selected}:
        raise ValueError('Select the fitted model for exactly one experimental series')
    profile = phase_runtime.load_physics_profile(parent)
    if profile is None or {key.split('/')[0] for key in profile['runs']} != {selected}:
        raise ValueError('The selected parent needs registered physical geometry for this series')
    manifest = update.read(parent / 'review_source' / 'postflight_manifest.json')
    jobs = manifest.get('jobs') or []
    names = [str(job['run']).casefold() for job in jobs]
    if (not jobs or len(set(names)) != len(names)
            or {series_id(job['dataset']) for job in jobs} != {selected}):
        raise ValueError('The parent must have one unambiguous registered review timeline for this series')
    if photometry_revision_root is not None:
        if selected != 'ffac':
            raise ValueError('The photometry source revision currently supports FF_AC/AC47 only')
        from darsia.fluidflower.tools.phase_training_photometry_revision import verify_revision
        ac47 = [job for job in jobs if str(job['run']).casefold() == 'ac47']
        if len(ac47) != 1:
            raise ValueError('The parent must have exactly one AC47 review source')
        photometry_revision_root = verify_revision(photometry_revision_root,
            parent_job=ac47[0])['root']
    if selected == 'ffbm' and set(names) != {'c1', 'c2', 'c3', 'c4', 'c5'}:
        raise ValueError('FF_BM retraining requires the complete C1–C5 parent, not a single-experiment pilot')
    snapshot = phase_review_retractions.load_retractions(series=selected)
    phase_review_retractions.assert_calibration_can_be_inherited(parent, snapshot)
    token = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8]
    output = library / f'phase-model-{token}-{selected}-unified'
    job = series_path(selected, 'results') / f'studio-review-update-{token}'
    if output.exists() or job.exists():
        raise FileExistsError('The newly allocated model or job folder already exists')
    args = SimpleNamespace(parent=parent, output=output, job=job, cache=cache/'studio-updates'/token,
        series=selected, actor=actor, backend=backend, cpus=cpus, memory_gb=memory_gb,
        feature_cache_root=(Path(feature_cache_root).resolve() if feature_cache_root is not None
                            else local_cache_root()/'regional_image_features'/selected),
        photometry_revision_root=photometry_revision_root,
        required_review=None, required_contributing_regions=None, capture_only=False,
        verify_numerics=False, eager_references=False, prepare_disagreements=False)
    return args


def progress_callback(args):
    """Emit the same event fields consumed by Studio's continuation parser."""
    def publish(stage, message, **extra):
        value = dict(stage=stage, model_id=args.output.name, output_root=str(args.output),
            request_path=str(args.output/'training_request.json'), job_root=str(args.job),
            series=args.series, message=message, **extra)
        print(EVENT_PREFIX + json.dumps(value, ensure_ascii=False), flush=True)

    def callback(value):
        stage = value['stage']
        if stage == 'preparing_studio':
            record_photometry_revision(args)
        if stage == 'failed':
            # Keep the last fitted/materializing event so Studio can correctly
            # report that a fitted model remains available after a map failure.
            print('Unified Studio training failed; see ' + str(args.job/'status.json'), flush=True)
            return
        mapping = {
            'capturing_reviews': ('capturing', 'Collecting saved corrections, blind labels and quiz answers from this model and its ancestors...'),
            'preflight_ready': ('prepared', 'Evidence frozen. Preparing cumulative regional training...'),
            'extracting_training_samples': ('extracting', 'Extracting optical features from all effective reviewed regions...'),
            'fitting_regional_heads': ('fitting', 'Fitting the common regional CO2 and gas model...'),
            'trained': ('assembling', 'Regional fit completed. Assembling the new model and common output rules...'),
            'studio_ready': ('materializing', 'Native phase maps are prepared. Finishing Studio registration...'),
            'ready_for_review': ('ready', 'New model and native phase maps are ready for review. Disagreements remain available on request.'),
        }
        if stage == 'preparing_studio':
            publish('fitted', 'New regional model fitted and visible. Preparing native phase maps...')
            publish('materializing', 'Calculating native phase maps with the common AC/UM/BM output rules...')
        elif stage in mapping:
            event_stage, message = mapping[stage]
            publish(event_stage, message, **{key: value[key] for key in
                ('new_aqueous_regions', 'total_regions', 'frames') if key in value})
    return callback


def run(parent, library, cache, selected, actor, **options):
    args = prepare_launch(parent, library, cache, selected, actor, **options)
    args.job.mkdir(parents=True, exist_ok=False)
    request = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    request.update(method='unified-regional-training',
        evidence_scope='All eligible cumulative corrections, ordinary blind labels and quiz answers; withdrawals and explicit promotions applied by the existing capture pipeline',
        output_scope='All experiments in the selected registered parent review timeline',
        optional_work='Numerical parity diagnostics, movies, proposal generation and disagreements are not requested')
    update.write(args.job/'launch_request.json', request)
    args.status_callback = progress_callback(args)
    update.run(args)
    return args.output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('parent', 'library', 'cache'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--series', choices=('ffac', 'ffum', 'ffbm'), required=True)
    parser.add_argument('--actor', required=True)
    parser.add_argument('--backend', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--cpus', type=int)
    parser.add_argument('--memory-gb', type=float)
    parser.add_argument('--feature-cache-root', type=Path)
    parser.add_argument('--photometry-revision-root', type=Path,
                        help='Verified immutable AC47 corrected-review revision; FF_AC only')
    args = parser.parse_args(argv)
    run(args.parent, args.library, args.cache, args.series, args.actor,
        backend=args.backend, cpus=args.cpus, memory_gb=args.memory_gb,
        feature_cache_root=args.feature_cache_root,
        photometry_revision_root=args.photometry_revision_root)
    return 0


if __name__ == '__main__':
    if os.name != 'nt':
        os.setsid()
    raise SystemExit(main())
