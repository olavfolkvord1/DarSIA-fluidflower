"""One experiment per process; delegates all scientific calculations unchanged."""
from __future__ import annotations
from darsia.fluidflower.tools._paths import load_snapshot_module

from darsia.fluidflower.series_paths import local_cache_root
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import traceback
import time

from darsia.fluidflower.tools.phase_mass_overnight_data import read, write, sha


def retry_json_write(writer, path, value, *, attempts=12, sleep=time.sleep, on_retry=None):
    """Retry transient Windows replacement denial without changing JSON data."""
    for attempt in range(attempts):
        try:
            return writer(path, value)
        except PermissionError:
            if attempt+1 == attempts:raise
            if on_retry is not None:on_retry(path, attempt+1)
            sleep(min(.05*2**attempt, 1.0))


def mass(job_path):
    from darsia.fluidflower.tools import phase_studio_mass as adapter
    folder=Path(job_path).parent
    original=adapter.write
    retries=[]
    def record(path,attempt):retries.append(dict(path=str(path),attempt=attempt))
    adapter.write=lambda path,value:retry_json_write(original,path,value,on_retry=record)
    try:
        return adapter.run(folder/'request.json')
    finally:
        adapter.write=original
        retry_json_write(write,folder/'mass_execution.json',dict(
            operation='ordinary Studio mass calculation with bounded JSON write retries',
            wrapper_sha256=sha(__file__),retries=retries,scientific_methods_unchanged=True))


def mass_candidate(validation, job):
    """Adapt production maps without requiring optional Studio movie/plot files."""
    from darsia.fluidflower.tools.calibration_pairwise_review import Candidate, _candidate_from_job
    if job.get('output_mode', 'review') == 'review':
        return _candidate_from_job(validation, job)
    if job.get('output_mode') != 'production':
        raise ValueError('Unknown phase output mode')
    def path(value):
        value = Path(value)
        return value if value.is_absolute() else Path(validation) / value
    folder = path(job['candidate_folder'])
    summary = read(folder / 'summary.json')
    if (summary.get('output_mode') != 'production'
            or summary.get('selection_candidate_id') != job['candidate_id']
            or summary.get('params_sha256') != job['params_sha256']):
        raise ValueError('Production mass metadata identifies another phase candidate')
    return Candidate(run=job['run'], role=job['role'], family=job.get('family', 'unknown'),
        variant=job['variant'], candidate_id=job['candidate_id'], grid_order=job.get('grid_order', 1),
        params_sha256=job['params_sha256'], normalized_mae=None,
        # Reserved paths are deliberately absent in production. The mass
        # adapter consumes the native photos, final maps and timeline instead.
        mass_plot=folder/'calibration_mass_fit.png', contour_movie=folder/'calibration_review.mp4',
        per_frame_csv=folder/'calibration_review_per_frame.csv', summary_path=folder/'summary.json',
        plot_seconds=0., image_seconds=0., fps=0., frame_count=int(summary['frame_count']),
        quality_flags=job.get('quality_flags', {}), dataset=job['dataset'],
        review_output_dir=path(job['review_output_dir']), phase_map_output_dir=path(job['phase_map_output_dir']),
        phase_model_only=True, optical_model_sha256=job['optical_model_sha256'],
        presence_diagnostics_dir=path(job['presence_diagnostics_dir']) if job.get('presence_diagnostics_dir') else None,
        phase_support_format=job.get('phase_support_format', ''))


def phase(job_path, *, output_mode=None):
    import cv2
    from threadpoolctl import threadpool_limits
    from darsia.fluidflower.tools import phase_transfer_review as transfer
    job_path=Path(job_path); job=read(job_path); folder=job_path.parent
    output_mode = output_mode or job.get('output_mode', 'review')
    if output_mode not in ('review', 'production'):
        raise ValueError('Phase output mode must be review or production')
    def update(message, **extra):
        write(folder/'phase_status.json',dict(stage=message,pid=os.getpid(),updated_utc=datetime.now(timezone.utc).isoformat(),**extra))
        print(message,flush=True)
    try:
        plan_path=folder/'transfer_plan.json'
        if plan_path.exists():plan=read(plan_path)
        else:
            plan=transfer.prepare_transfer_review(Path(job['model_root']),job['run'],Path(job['source_manifest']),
                Path(job['transfer_root']),physics_mode='prepared-target',review_mode='visible')
            write(plan_path,plan)
        wrapper_path=Path(job['resource_wrapper'])
        wrapper = load_snapshot_module('campaign_resources', wrapper_path)
        from darsia.fluidflower.tools import materialize_phase_iteration as inference
        inference.CudaFeaturePredictor=wrapper.TransientCudaFeaturePredictor
        threads=int(job.get('threads',2));cv2.setNumThreads(threads)
        with threadpool_limits(limits=threads):
            binding=transfer.materialize_transfer_review(plan,progress=update,backend=job.get('backend','cuda'),
                output_mode=output_mode)
        root=Path(plan['evaluation_root']); validation=root/'validation'
        campaign=read(validation/'postflight_manifest.json')
        candidate=mass_candidate(validation,campaign['jobs'][0])
        request=dict(review_root=str(candidate.review_output_dir),candidate=asdict(candidate),frame_index=0,
            reference_mode=False,local_baseline_search=False,method=job['method'],analysis_width=1024,
            results_root=str(folder/'results'),phase_output_mode=output_mode)
        write(folder/'request.json',request)
        # Use the normal Studio transfer discovery receipt and normal commands.
        command=(transfer.visible_review_command(plan,binding,reviewer='Olav Folkvord',
            save_root=Path(job['review_save_root']),cache_root=local_cache_root())
            if output_mode == 'review' else None)
        result=dict(stage='ready',evaluation_root=str(root),binding=binding,
            completed_utc=datetime.now(timezone.utc).isoformat(),model_root=job['model_root'],
            model_id=plan['source_model_id'],source_model_sha256=plan['source_model_sha256'],
            target_run=plan['run'],target_native_manifest_sha256=sha(plan['target_native_manifest']),
            command=command,reopen_command=command,correction_command=None,review_mode='visible',
            output_mode=output_mode,
            **{k:plan[k] for k in ('training_seen','label','physics_mode','optical_training_status',
                'calibration_evidence_status','exposure_label','calibration_history')})
        write(root/('transfer_result.json' if output_mode == 'review' else 'production_result.json'),result)
        write(folder/'resource_receipt.json',dict(wrapper=str(wrapper_path),sha256=sha(wrapper_path),
            events=wrapper.EVENTS,threads=threads,output_mode=output_mode,scientific_methods_unchanged=True))
        update('complete',request=str(folder/'request.json'),evaluation_root=str(root))
    except Exception:
        update('failed',error=traceback.format_exc());raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job',type=Path,required=True)
    parser.add_argument('--operation',choices=['phase','mass'],default='phase')
    parser.add_argument('--output-mode',choices=['review','production'],
        help='Phase artifacts to generate; defaults to job output_mode or review.')
    args=parser.parse_args()
    if args.operation=='mass':
        mass(args.job)
    else:
        phase(args.job,output_mode=args.output_mode)
