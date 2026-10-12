"""Prepare a sparse FF_UM timeline from its own rig, photos and protocols.

No phase weights, reviewed regions, colour-path fit or mass fit are loaded.
The native JPEG contract is the same one used by Studio and its mass adapter.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import shutil
import threading
import time

import cv2
import numpy as np

from darsia.fluidflower.tools.prepare_phase_injection_references import um_reference_source
from darsia.fluidflower.tools.phase_mass_overnight_data import read, write, sha

TIMES = (.25, .5, .75, 1., 1.25, 1.75, 2., 2.25, 2.75, 3., 4., 5., 6., 8., 10., 12., 24., 48., 72., 96.)


def encode_native_jpeg(rgb, manifest, run, elapsed_h=0.):
    """The encoding half of ``concentration_movie_compute.encode_review_rgb``: identical bytes.

    That function also decodes the 51 MP JPEG again to return its pixels, which the preparation
    never used. ``tests/test_prepare_ffum_phase_target_encoding.py`` keeps the two in step.
    """
    from darsia.fluidflower.tools import schema2_phase_transfer_campaign as transfer
    width, height = manifest["canvas_wh"]
    canvas = transfer._plain_frame(np.clip(np.asarray(rgb, np.float32), 0, 1),
        run=run, index=0, count=1, elapsed_h=elapsed_h,
        width=width, height=height, native=True)
    success, encoded = cv2.imencode(".jpg", cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR),
                                   [int(cv2.IMWRITE_JPEG_QUALITY), 98])
    if not success:
        raise RuntimeError("Native corrected JPEG encoding failed")
    return encoded


def array_sha256(array):
    """``schema2_phase_transfer_campaign._array_sha256`` without its 447 MB ``tobytes()`` copy."""
    return hashlib.sha256(np.ascontiguousarray(np.asarray(array, dtype=np.float32))).hexdigest()


PREPARATION_THREADS_ENV = "PHASE_PREPARATION_THREADS"


def preparation_threads():
    """Photographs corrected at once; 4 was verified bit-identical on the workstation, 1 is the old loop."""
    value = os.environ.get(PREPARATION_THREADS_ENV, "").strip()
    return max(1, int(value)) if value else 4


def correct_all(frames, correct, threads):
    """The baseline alone first, then the photographs several at a time; results stay in frame order."""
    results = [correct(frames[0])]
    rest = list(frames[1:])
    if threads <= 1 or len(rest) < 2:
        return results + [correct(row) for row in rest]
    with ThreadPoolExecutor(threads, thread_name_prefix="prepare") as pool:
        return results + list(pool.map(correct, rest))      # list() re-raises the first failure


def recheck_concurrent_correction(row, read_corrected, threads):
    """Correct one concurrently corrected photograph again, alone; a different hash stops the preparation."""
    again = array_sha256(read_corrected(Path(row['source_path'])))
    if again != row['corrected_array_sha256']:
        raise RuntimeError(f"Photograph {row['source_name']} corrected beside {threads - 1} others differs from its "
                           f"correction alone; set {PREPARATION_THREADS_ENV}=1 and prepare again")
    return dict(threads=threads, rechecked_frame=int(row['frame_index']), identical_to_correction_alone=True)


def prepare(run, output, dataset='ffum'):
    import darsia
    from darsia.fluidflower.tools import schema2_phase_transfer_campaign as shared
    from darsia.fluidflower.tools.concentration_movie_compute import encode_review_rgb, save_npz
    from darsia.fluidflower.tools.phase_injection_points import project_injection_points
    from darsia.fluidflower.phase_plume import PlumeContext
    from darsia.fluidflower.tools.prepare_phase_transfer_target import coordinate_arrays
    from threadpoolctl import threadpool_limits
    output = Path(output).resolve()
    manifest_path = output/'postflight_manifest.json'
    if manifest_path.is_file(): return read(output/'preparation_report.json')
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    status = output/'status.json'
    def update(message):
        write(status, dict(stage=message, updated_utc=datetime.now(timezone.utc).isoformat()))
        print(run+': '+message, flush=True)
    cv2.setNumThreads(2)
    shared.optical_control.install_runtime_hooks()
    if dataset not in {'ffum','ffac'}: raise ValueError('Unsupported registered target series')
    manager = (um_reference_source(shared, output, run) if dataset=='ffum'
               else shared._open_ffac(run, require_clearwater=False))
    with threadpool_limits(limits=2), manager as source:
        update('Checking target rig, clear-water reference and sparse acquisition times')
        rig, experiment = source.rig, source.experiment
        baseline = np.asarray(source.baseline, np.float32)
        baseline_path = Path(rig.baseline_path)
        baseline_time = float(experiment.time_since_start(experiment.get_datetime(baseline_path)))
        if baseline_time >= 0: raise ValueError('Configured baseline is not before injection')
        chosen = shared._select_frames(experiment, TIMES, tolerance_seconds=None)
        frames = [dict(frame_index=0, elapsed_h=baseline_time, source_name=baseline_path.name,
            source_path=str(baseline_path), selection_status='pre_injection_negative_control')]
        for frame in chosen:
            if frame.elapsed_h <= 0 or frame.path == baseline_path: continue
            if frame.elapsed_h <= frames[-1]['elapsed_h']: raise ValueError('Nonmonotonic target timeline')
            frames.append(dict(frame_index=len(frames), elapsed_h=frame.elapsed_h, source_name=frame.path.name,
                source_path=str(frame.path), selection_status='nearest_available_acquisition'))
        stages = shared._source_injection_stages(source)
        if dataset=='ffum':
            rig_path = Path(source.correction_contract.get('rig_path') or source.correction_contract['legacy_rig'])
        else:
            from darsia.presets.workflows.config.fluidflower_config import FluidFlowerConfig
            rig_path = FluidFlowerConfig(source.correction_contract['config_paths'],require_data=True,require_results=False).rig.path
        facies, geometry = np.asarray(rig.facies.img), rig.facies.metadata()
        x_m, y_m = coordinate_arrays(facies, geometry, baseline.shape)
        with np.load(rig_path/'permeability.npz') as archive:
            permeability = archive['array'].astype(np.float32)
        excluded = 1 if dataset=='ffum' else 0
        domain = facies != excluded
        context = PlumeContext(domain, facies, x_m, y_m, stages, permeability)
        clear_paths = (shared._ffum_clearwater_paths(run) if dataset=='ffum'
                       else shared.select_clear_reference_paths(shared.FFAC_RAW_ROOT,run))
        clear, clear_contract = shared._read_corrected_clearwater(rig, clear_paths, baseline.shape)
        if not np.isfinite(clear).all(): raise ValueError('Nonfinite clear-water correction')
        clearwater_fallback = shared.saturation_fallback.take_last_record()      # clipped chart patches, flagged
        if clearwater_fallback: clear_contract = dict(clear_contract, saturation_fallback=clearwater_fallback)
        # Keep a small reference pair for visual QA, as well as exact source hashes.
        for name, image in (('clearwater', clear), ('baseline', baseline)):
            h,w = image.shape[:2]
            small = cv2.resize(np.clip(image,0,1), (1000,round(h*1000/w)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(output/(name+'_preview.jpg')), cv2.cvtColor(np.rint(small*255).astype(np.uint8),cv2.COLOR_RGB2BGR))
        write(output/'clearwater_preflight.json',dict(contract=clear_contract,
            source_inputs=[dict(path=str(p),sha256=sha(p)) for p in clear_paths],
            finite=True,visual_review_status='pending',not_a_known_concentration=True))
        del clear
        context_path=output/'target_context'/f'{run}.npz'
        save_npz(context_path,domain=context.domain,facies=context.facies,x_m=x_m,y_m=y_m,permeability=permeability)
        width,height = shared.review._native_review_dimensions(baseline.shape)
        box = shared.review._native_content_box(baseline.shape,width,height)
        review=output/'manual_pairwise_review'
        native_path=review/'native_frames'/run/'manifest.json'
        native=dict(schema=1,dataset=dataset,run=run,kind='native_resolution_corrected_review_frames',
            frame_count=len(frames),canvas_wh=[width,height],analysis_image_shape_hw=list(baseline.shape[:2]),
            content_box_xywh=list(box),full_analysis_resolution=True,additional_spatial_resampling=False,
            jpeg_quality=98,frame_sources=frames,correction_contract=source.correction_contract,
            domain=dict(path=str(context_path),sha256=sha(context_path),excluded_facies=[excluded]),
            phase_topology=dict(injection_stages=list(stages),previous_predictions_used=False),
            reference=dict(kind='radiometrically_corrected_pre_injection_baseline',source_path=str(baseline_path),
                source_name=baseline_path.name,source_sha256=sha(baseline_path),elapsed_h=baseline_time,
                included_in_timeline=True,timeline_frame_index=0,content_box_xywh=list(box)),
            injection_points=project_injection_points(stages,x_m=x_m,y_m=y_m,canvas_wh=[width,height],content_box=box),
            phase_evaluation_domain=dict(name='registered_porous_medium_2d',continuous_image_porosity_used_as_hard_mask=False),
            created_utc=datetime.now(timezone.utc).isoformat())
        # Laps are kept OUT of ``row``: the rows are part of the hashed native manifest.
        context_s=time.perf_counter()-started
        status_lock=threading.Lock();finished=[0]
        def read_corrected(path):
            return np.asarray(darsia.imread(path,transformations=rig.corrections,
                reference_date=rig.reference_date,name=path.name,dtype=np.float32).img,np.float32)[...,:3]
        def correct(row):
            i=row['frame_index'];path=Path(row['source_path'])
            clock=[time.perf_counter()];lap={}
            def mark(name):
                now=time.perf_counter();lap[name]=round(now-clock[0],3);clock[0]=now
            row['source_sha256']=sha(path);mark('source_hash_s')
            rgb=baseline if i==0 else read_corrected(path)
            mark('read_and_correct_s')
            fallback=shared.saturation_fallback.take_last_record()   # this thread's frame: clipped chart patches?
            if rgb.shape!=baseline.shape or not np.isfinite(rgb).all(): raise ValueError('Invalid corrected image')
            mark('validate_s')
            encoded=encode_native_jpeg(rgb,native,run,row['elapsed_h']);mark('encode_s')
            target=native_path.parent/f'frame_{i:03d}.jpg';target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(encoded.tobytes());row['native_jpeg_sha256']=sha(target);mark('write_and_hash_jpeg_s')
            row['corrected_array_sha256']=array_sha256(rgb);mark('corrected_array_hash_s')
            if i==0:
                reference=review/'native_references'/run/'pre_injection.jpg';reference.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(target,reference)
            if sha(path)!=row['source_sha256']: raise ValueError('Raw photograph changed during preparation')
            mark('source_recheck_s')
            with status_lock:                        # status.json is replaced through one temporary file
                finished[0]+=1;update(f'Correcting native photograph {finished[0]}/{len(frames)}')
            return dict(frame_index=i,source_name=path.name,**lap,total_s=round(sum(lap.values()),3),
                        **({'saturation_fallback':fallback} if fallback else {}))
        from darsia.fluidflower.tools import phase_memory_defaults
        with phase_memory_defaults.scoped() as memory_defaults:      # a small machine corrects one photograph at a time
            threads=preparation_threads()
        photographs_started=time.perf_counter()
        frame_timings=correct_all(frames,correct,threads)
        photographs_s=time.perf_counter()-photographs_started
        concurrency=dict(threads=1)
        if threads>1 and len(frames)>2:
            concurrency=recheck_concurrent_correction(frames[1],read_corrected,threads)
        write(native_path,native)
        provenance=dict(target_binding=dict(dataset=dataset,run=run,native_manifest_sha256=sha(native_path),
            source_manifest=str(manifest_path)),rig_source=str(rig_path),correction_contract=source.correction_contract,
            geometry=dict(indexing='ij',dimensions=list(map(float,geometry['dimensions'])),origin=list(map(float,geometry['origin']))),
            training_labels_used=False,calibration_fitted=False,current_phase_predictions_used=False,
            reference_manifest_sha256=sha(output/'clearwater_preflight.json'))
        descriptor=dict(format='phase-transfer-target-context-v1',dataset=dataset,run=run,
            native_manifest_sha256=sha(native_path),path=str(context_path),sha256=sha(context_path),
            injection_stages=list(stages),facies_identity='same-series-categorical-labels',provenance=provenance)
        job=dict(job_id=f'{dataset}_{run}_prepared',dataset=dataset,run=run,variant='prepared_target',
            candidate_folder=str(output/'prepared_inputs'/run),review_output_dir=str(review),
            analysis_source_review_root=str(review),phase_transfer_context=descriptor)
        (output/'prepared_inputs'/run).mkdir(parents=True,exist_ok=True)
        report=dict(status='prepared_target_inputs_ready',run=run,manifest=str(manifest_path),native_manifest=str(native_path),
            frame_count=len(frames),time_range_h=[frames[0]['elapsed_h'],frames[-1]['elapsed_h']],
            training_performed=False,phase_inference_performed=False,
            radiometry=dict(saturation_fallback=shared.saturation_fallback.report(limit=50)),
            timings=dict(note='Wall-clock laps; frame 0 is the baseline corrected while the context was built.',
                context_before_first_photo_s=round(context_s,3),frames=frame_timings,
                all_photographs_wall_s=round(photographs_s,3),concurrent_correction=concurrency,
                memory_defaults=memory_defaults,
                total_s=round(time.perf_counter()-started,3)))
        write(output/'preparation_report.json',report)
        write(manifest_path,dict(schema=1,dataset=dataset,kind='label-free-prepared-phase-transfer-target',
            selection_id=output.name,include_in_training=False,jobs=[job]))
        update('complete')
        return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--dataset',choices=['ffum','ffac'],default='ffum');args=parser.parse_args()
    prepare(args.run,args.output,args.dataset)
