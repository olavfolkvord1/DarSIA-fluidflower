"""Root-only CPU 127-frame precision test; no production permission.

No work happens on import. Root supplies immutable manifests, staged identities,
explicit SCI-extension source review and fresh original controls. Each worker
owns one whole AC run: chronological float64 history, then float32 history.
Neither an old IO-only review nor an old four-frame result is a 127 proof.
"""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
import uuid
from types import FunctionType, ModuleType

COUNTS = {'ac18':44,'ac52':39,'ac19':44}
SPEC_SHA = '36be522e1323c6bf376f0868e3ab52a21090185d6908322d7c48ed450df2ed96'
PASS_SHA = 'a8e7971f0ee8ea8a2b34f34a7282e029dc26511e0837a8332e8593dfa1fda2d4'
B1_SHA = '38f8d821eabf92d25ceb8a6ce76f69d7c2054fb888fbab9cac6bdb03e2f5f00c'
THREAD_ENV = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
GIB = 1024**3


def _bound(value):
    if (not isinstance(value,Mapping) or set(value)!={'path','sha256','size_bytes'}
            or not isinstance(value['path'],str) or not Path(value['path']).is_absolute()
            or re.fullmatch('[0-9a-f]{64}',value.get('sha256','')) is None
            or type(value.get('size_bytes')) is not int or value['size_bytes']<0):
        raise ValueError('One exact absolute SHA/integer-size binding required')
    return dict(value)


def descriptor_equal(a,b,normalize):
    a,b=_bound(a),_bound(b)
    return (normalize(a['path'])==normalize(b['path']) and a['sha256']==b['sha256'] and a['size_bytes']==b['size_bytes'])


def load_source(name, bound):
    """Verify external prototype code before import; never call numerical code."""
    bound=_bound(bound);path=Path(bound['path'])
    if path.stat().st_size!=bound['size_bytes'] or hashlib.sha256(path.read_bytes()).hexdigest()!=bound['sha256']:
        raise ValueError('Prototype source bytes differ: '+name)
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def load_baseline(sources):
    """Load only the stdlib IO/job/input modules before worker handshake."""
    package=ModuleType('_precision_benchmark_baseline');package.__path__=[]
    sys.modules[package.__name__]=package
    result={}
    for name in ('production_io','production_job','production_inputs'):
        result[name]=load_source(package.__name__+'.'+name,sources[name])
    return result


def _clone(function, **overrides):
    value=FunctionType(function.__code__,dict(function.__globals__,**overrides),
        function.__name__,function.__defaults__,function.__closure__)
    value.__kwdefaults__=function.__kwdefaults__
    return value


class _Proxy:
    def __init__(self, original, **overrides):self.original,self.overrides=original,overrides
    def __getattr__(self,key):return self.overrides[key] if key in self.overrides else getattr(self.original,key)


def validate_metadata_packet(packet, *, selection, closure):
    """Metadata shape only, before staging/worker/context/array operations."""
    if (packet.get('format')!='darsia-production-cpu-precision127-packet-v1'
            or packet.get('B1_test_only') is not True or packet.get('production_permission_granted') is not False
            or packet.get('host')!='Main' or packet.get('backend')!='cpu' or packet.get('worker_count')!=3
            or packet.get('memory_floor_bytes')!=20*GIB or packet.get('reserve_bytes')!=40*GIB
            or packet.get('worker_budget_bytes')!=20*GIB or packet.get('limits',{}).get('sha256')!=SPEC_SHA
            or packet.get('selected_calibration',{}).get('verifier')!='round3'
            or packet['selected_calibration']['asset'].get('sha256')!=B1_SHA):
        raise ValueError('Main three-worker B1-only fixed20/40 CPU precision test required')
    if (type(packet.get('threads')) is not int or packet['threads']<=0
            or type(packet.get('logical_cpus')) is not int or packet['threads']*3>packet['logical_cpus']):
        raise ValueError('Actual thread/worker/logical CPU admission differs')
    if (selection.get('format')!='darsia-production-precision127-selection-v1'
            or selection.get('complete') is not True or selection.get('run_counts')!=COUNTS
            or selection.get('case_count')!=127 or selection.get('limits',{}).get('sha256')!=SPEC_SHA
            or closure.get('format')!='darsia-production-precision127-flat-closure-v1'
            or closure.get('complete') is not True or closure.get('run_counts')!=COUNTS
            or closure.get('scientific_arrays_read') is not False):
        raise ValueError('Exact original127 metadata selection/closure required')
    cases={row['case_id']:row for row in selection['cases']}
    if len(cases)!=127 or set(packet['runs'])!=set(COUNTS):
        raise ValueError('Exact three whole-run ownership rows required')
    owners=set()
    for run,count in COUNTS.items():
        row=packet['runs'][run]
        if row.get('host')!='Main' or type(row.get('worker')) is not int or row['worker'] in owners:
            raise ValueError('One unique Main worker per whole AC run required')
        owners.add(row['worker'])
        expected=sorted((value for value in cases.values() if value['entry']['run']==run),
                        key=lambda value:(value['entry']['elapsed_h'],value['entry']['frame_index']))
        if len(expected)!=count or row['case_ids']!=[v['case_id'] for v in expected]:
            raise ValueError('Whole-run frame order differs from locked chronological selection')
        for precision in ('float64','float32'):
            for key in ('manifest','verified_stage','configuration'):_bound(row[precision][key])
    if len(owners)!=3 or set(packet['sources'])!={'production_io','production_job','production_inputs','production_native','production_retained',
            'stream_io','lifecycle','kernel','adapter','selection','comparator','runner'}:
        raise ValueError('All actual baseline and new SCI extension source bindings required')
    for value in packet['sources'].values():_bound(value)
    _bound(packet['source_review'])
    return cases


def extension_reader_class(base, stream, *, source_review, executed_sources, run, io):
    """Honest SCI-test qualification, not old IO32 parity or production eligibility.

    Original constructor still checks clean commit/environment/restart, logical
    inputs, aliases, source provenance and opaque evidence. Source path lookup,
    all source byte checks and native input admission stay original.
    """
    class ScienceReviewReader(base):
        def _qualify_sources(self, manifest, qualification):
            if (not isinstance(qualification,Mapping) or set(qualification)!={'source_review'}
                    or not self._same_descriptor(qualification['source_review'],source_review)):
                raise ValueError('Exact external Root SCI-test source review required')
            review=io.read_json(source_review,max_bytes=source_review['size_bytes'])
            science=manifest['scientific_input_roles'];assurance=manifest['assurance_input_roles']
            if (set(science)&set(assurance) or set(science)|set(assurance)!=set(manifest['inputs'])
                    or not any(self._same_descriptor(source_review,manifest['inputs'][r]) for r in assurance)):
                raise ValueError('SCI review must be separate declared assurance, not a scientific input')
            if (review.get('contract')!='darsia-production-precision127-source-review-v1'
                    or review.get('complete') is not True or review.get('approved_for_cpu_precision_test') is not True
                    or review.get('B1_test_only') is not True or review.get('production_permission_granted') is not False
                    or review.get('execution_commit')!=self.code_commit
                    or review.get('settings')!={'host':'Main','backend':'cpu','precisions':['float64','float32']}
                    or review.get('original_native_guard_code_objects_preserved') is not True
                    or review.get('original_all_39_plan_guards_preserved') is not True
                    or not self._same_input_bindings(review['input_bindings_by_run'][run],{r:manifest['inputs'][r] for r in science})
                    or not self._same_source_bindings(review['source_bindings'],self.source_bindings)
                    or not self._same_input_bindings(review['executed_sources'],executed_sources)
                    or manifest['purpose']!='equivalence' or manifest['settings']['host']!='Main'
                    or manifest['settings']['backend']!='cpu' or manifest['production_permission_granted'] is not False):
                raise ValueError('Root-reviewed SCI-extension/input/source/precision scope differs')
            self._assurance_locators={self._canonical_key(manifest['inputs'][r]['path']) for r in assurance}
            for row in self.source_bindings:
                self.io.verify(row['current_source']);self.path(row['original_source'])
            self.source_qualification_state='source_reviewed_CPU_precision127_test_only'
            self.source_qualification=deepcopy(qualification)
    stage_class=stream.stream_reader_class(ScienceReviewReader)
    class PrefetchedScienceReader(stage_class):
        def array(self,descriptor):
            cache=getattr(self,'_prefetched',{})
            logical=self.logical_descriptor(descriptor);bound=self._stage_logical(descriptor)
            key=self._canonical_key(bound['path'])
            if key not in cache:return super().array(descriptor)
            if ((logical['sha256'],logical['size_bytes']) in self._opaque_contents
                    or self._canonical_key(logical['path']) in self._assurance_locators):
                raise ValueError('Opaque/assurance evidence cannot enter prefetched arrays')
            if self._frame_stack is None:raise ValueError('Prefetch use requires the guarded current frame interval')
            import numpy as np
            value=cache[key];contract=self._array_contracts[key]
            self.path(bound)  # Fresh pinned-local identity and actual worker reference.
            if (tuple(value.shape)!=contract['shape'] or value.dtype!=np.dtype(contract['dtype'])
                    or value.flags.writeable or value.dtype.hasobject):
                raise ValueError('Prefetched array differs from the exact immutable NPY contract')
            return value
    return PrefetchedScienceReader


def candidate_gate(native, manifest, source_review, run):
    """Separate test-only gate; original native prepare/load codeobjects retained."""
    def gate(configuration, reader):
        if (configuration['execution_mode']!='candidate_io' or configuration['purpose']!='file_handling_acceptance'
                or configuration['production_permission_granted'] is not False
                or reader.source_qualification_state!='source_reviewed_CPU_precision127_test_only'
                or reader.source_qualification['source_review']!=source_review
                or native._execution_identity(reader)!=reader.execution_identity
                or manifest['settings']['host']!='Main' or manifest['settings']['backend']!='cpu'
                or not callable(getattr(reader,'source_path',None)) or not reader.source_bindings):
            raise ValueError('Exact source-reviewed unqualified CPU precision test reader required')
        return reader
    return gate


def thread_facts(threads, *, cv2, poolctl):
    pools=poolctl.threadpool_info()
    env={key:os.environ.get(key) for key in THREAD_ENV}
    if any(value!=str(threads) for value in env.values()) or cv2.getNumThreads()!=threads:
        raise RuntimeError('Actual requested environment/OpenCV threads differ')
    if any(row.get('num_threads')!=threads for row in pools):
        raise RuntimeError('Actual native BLAS/OpenMP thread pools differ')
    return dict(environment=env,opencv_threads=cv2.getNumThreads(),pools=pools)


def status_data(path):
    """Fresh status DATA only; no scientific descriptor/timestamp coercion.

    Operational float clocks may occur elsewhere in a status document. Native
    worker epochs are validated separately by their exact string/PID binding.
    Duplicate keys, nonfinite values, oversized or changing documents fail.
    """
    def pairs(rows):
        result={}
        for key,value in rows:
            if key in result:raise ValueError('Duplicate status DATA key')
            result[key]=value
        return result
    def nonfinite(value):raise ValueError('Nonfinite status DATA value')
    def number(value):
        import math
        result=float(value)
        if not math.isfinite(result):raise ValueError('Nonfinite status DATA number')
        return result
    path=Path(path);before=path.stat()
    with path.open('rb') as stream:raw=stream.read(16*1024**2+1)
    after=path.stat()
    if (len(raw)>16*1024**2 or len(raw)!=before.st_size
            or (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)!=
               (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)):
        raise ValueError('Status DATA changed or exceeded its bounded read')
    value=json.loads(raw.decode('utf-8-sig'),object_pairs_hook=pairs,parse_constant=nonfinite,parse_float=number)
    if not isinstance(value,dict):raise ValueError('Status DATA must be an object')
    return value


@contextmanager
def local_fence_mutex(packet_sha256):
    """Local kernel synchronization of this packet's control read/replace only.

    No NAS lock file, queue, permission retry or cached authority. The original
    NasIO strict read/atomic replacement stays unchanged inside this boundary.
    Abandoned ownership still proceeds through every original content check.
    """
    if re.fullmatch('[0-9a-f]{64}',packet_sha256 or '') is None:
        raise ValueError('Exact immutable packet SHA required for local control mutex')
    if os.name!='nt':raise RuntimeError('This Main control protocol requires Windows named mutex support')
    import ctypes
    from ctypes import wintypes
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    create=kernel.CreateMutexW;create.argtypes=(wintypes.LPVOID,wintypes.BOOL,wintypes.LPCWSTR);create.restype=wintypes.HANDLE
    wait=kernel.WaitForSingleObject;wait.argtypes=(wintypes.HANDLE,wintypes.DWORD);wait.restype=wintypes.DWORD
    release=kernel.ReleaseMutex;release.argtypes=(wintypes.HANDLE,);release.restype=wintypes.BOOL
    close=kernel.CloseHandle;close.argtypes=(wintypes.HANDLE,);close.restype=wintypes.BOOL
    handle=create(None,False,'Local\\FFAC_'+packet_sha256)
    if not handle:raise ctypes.WinError(ctypes.get_last_error())
    acquired=False
    try:
        outcome=wait(handle,5000)
        if outcome==258:raise TimeoutError('Local control mutex exceeded its fixed5second wait')
        if outcome not in (0,128):raise ctypes.WinError(ctypes.get_last_error())
        acquired=True
        yield
    finally:
        release_error=0
        if acquired and not release(handle):release_error=ctypes.get_last_error()
        close_error=0 if close(handle) else ctypes.get_last_error()
        if release_error or close_error:raise ctypes.WinError(release_error or close_error)


def guard(packet, io, identity, *, job_module, psutil, threads=None, cv2=None, poolctl=None, check_coordinator=True):
    """Existing thread-worker PASS/noCONCERN guard with explicit new test scope."""
    claude=status_data(packet['claude_status_path']);status=status_data(packet['codex_status_path'])
    if (claude['label_check']['state']!='PASS' or claude['label_check']['manifest_sha256']!=PASS_SHA
            or any(any(row.get(k)=='CONCERN' for k in ('review','verdict','state')) for row in claude.get('technical_fix_review',[]))):
        raise RuntimeError('Original phase PASS/noCONCERN guard denies current frame')
    prep=status['production_preparation']
    if any(prep.get(key)!=value for key,value in packet['authorization'].items()):
        raise RuntimeError('Current direct B1 optimization/distribution scope changed')
    directive=prep['next_optimization_directive'];distribution=prep['distribution_direct_instruction']
    if (directive.get('B1_test_only') is not True or directive.get('production_go') is not False
            or distribution.get('B1_test_only') is not True or distribution.get('production_go') is not False
            or distribution.get('whole_experiment_one_host') is not True
            or distribution.get('whole_experiment_one_worker') is not True
            or distribution.get('NAS_shared_queue') is not False or distribution.get('NAS_lock_files') is not False):
        raise RuntimeError('B1 test authorization is not a production GO')
    if job_module.worker_identity()!=identity:
        raise RuntimeError('Actual native PID/start time differs')
    if psutil.virtual_memory().available<=40*GIB or psutil.Process().memory_info().peak_wset>packet['worker_budget_bytes']:
        raise RuntimeError('Actual memory exceeds fixed20 GiB worker/40 GiB reserve')
    for source in packet['executed_source_assets']:io.verify(source)
    if check_coordinator and packet.get('coordinator_fence'):
        with local_fence_mutex(packet['packet_sha256']):
            fence=io.read_json(packet['coordinator_fence'])
        age=(datetime.now(timezone.utc)-datetime.fromisoformat(fence['updated_utc'])).total_seconds()
        if (age<0 or age>10 or fence.get('worker_packet_sha256')!=packet['packet_sha256']
                or fence.get('continue') is not True):
            raise RuntimeError('Fresh local coordinator ownership/freshness denies the frame')
        parent=psutil.Process(fence['owner']['pid'])
        if str(parent.create_time())!=fence['owner']['process_create_time']:
            raise RuntimeError('Actual local coordinator identity changed')
    if threads is not None:thread_facts(threads,cv2=cv2,poolctl=poolctl)


def _anchor(reader, descriptor, context, run):
    value=reader.json(descriptor)
    entry=next(row['anchor_entry'] for row in context.cases.values() if row['entry']['run']==run)
    if (value.get('complete') is not True or value['campaign_sha256']!=context.configuration['campaign']['sha256']
            or value['run']!=run or value['frame_index']!=entry['frame_index'] or value['selected_target_h']!=2.25
            or value['elapsed_h']!=entry['elapsed_h']):
        raise ValueError('Actual saved original2.25h anchor differs')
    anchor=value['evaluation']['anchor']
    if (anchor['frame_index']!=entry['frame_index'] or anchor['elapsed_h']!=entry['elapsed_h']
            or anchor['affected_pore_volume_m3']<=0 or anchor['affected_volume_imputes_missing_geometry'] is not False):
        raise ValueError('Original native affected-volume anchor differs')
    return anchor


def _prefetch_plan(context, reader, case_id):
    """One bounded next-frame working input; no speculative dependency traversal."""
    phase=reader.json(context.cases[case_id]['entry']['receipt'])
    descriptor=phase['full_extended_scientific'];result={}
    if 'native_centre' in descriptor:
        result['centre']=descriptor['native_centre']['log_linear_rgb']
        result['centre_valid']=descriptor['native_centre']['valid_mask']
    if Path(phase['labels_map']['path']).suffix.lower()=='.npy':result['labels']=phase['labels_map']
    # Ring windows may be many arrays, so the bounded prefetch initially warms
    # the first exact ring only. Native loader still reads ALL original windows.
    if descriptor.get('ring_windows'):result['ring']=descriptor['ring_windows'][0]['log_linear_rgb']
    return result


def metadata_blueprint(request, *, io):
    """Fixed logical manifest tables for Root's source review, no payload reads."""
    preparation=io.read_json(request['metadata_preparation'],max_bytes=request['metadata_preparation']['size_bytes'])
    selection=io.read_json(preparation['selection'],max_bytes=preparation['selection']['size_bytes'])
    closure=io.read_json(preparation['flat_closure'],max_bytes=preparation['flat_closure']['size_bytes'])
    if (preparation.get('complete') is not True or preparation.get('scientific_payload_bytes_verified') is not False
            or preparation.get('run_counts')!=COUNTS or len(closure['assets'])!=5481
            or request.get('B1_test_only') is not True or request.get('production_permission_granted') is not False):
        raise ValueError('Actual original127 metadata preparation and B1-only request required')
    import ntpath
    io_normalize=lambda value:ntpath.normcase(ntpath.normpath(ntpath.abspath(value)))
    inputs={};aliases={io_normalize(key):_bound(value) for key,value in preparation['logical_aliases'].items()}
    def canonical(bound):
        original=_bound(bound);target=aliases.get(io_normalize(original['path']))
        if target is None:return original
        if (target['sha256'],target['size_bytes'])!=(original['sha256'],original['size_bytes']):
            raise ValueError('Pre-existing logical alias changes frozen SHA/integer size')
        return deepcopy(target)
    for run in COUNTS:
        # Full flat admission remains global: the immutable5481 union is bound
        # for every worker. Working ownership/deletion is still per one AC run.
        rows={f'asset_{index:05d}':canonical(row['descriptor']) for index,row in enumerate(closure['assets'])}
        rows.update(native_configuration=_bound(preparation['native_configuration']),
            original_anchor_2p25=_bound(request['runs'][run]['anchor_receipt']),boxes_common=_bound(request['boxes']))
        # Preserve every old alias target, including the old-four selection
        # required by the unchanged original-four guard before127 admission.
        declared={io_normalize(bound['path']):bound for bound in rows.values()}
        for index,target in enumerate(aliases.values()):
            key=io_normalize(target['path'])
            if key in declared:
                if not descriptor_equal(declared[key],target,io_normalize):
                    raise ValueError('Declared alias target conflicts with frozen canonical bytes')
            else:
                rows[f'preserved_alias_target_{index:03d}']=deepcopy(target);declared[key]=target
        inputs[run]=rows
    return dict(format='darsia-production-precision127-logical-blueprint-v1',complete=True,
        metadata_preparation=request['metadata_preparation'],selection=preparation['selection'],closure=preparation['flat_closure'],
        input_bindings_by_run=inputs,source_provenance=preparation['source_provenance'],
        logical_aliases=preparation['logical_aliases'],selected_calibration=preparation['selected_calibration'],
        case_ids_by_run={run:[row['case_id'] for row in selection['cases'] if row['entry']['run']==run] for run in COUNTS},
        source_review_pending=True,scientific_arrays_read=False,production_permission_granted=False,
        original_logical_closure=preparation['flat_closure'],canonical_targets_only_from_preexisting_aliases=True)


def source_review_template(request,blueprint,*,io):
    """Unapproved metadata for Root's actual source decision. Never an approval."""
    manifest=io.read_json(request['manifest_template'],max_bytes=request['manifest_template']['size_bytes'])
    return dict(contract='darsia-production-precision127-source-review-v1',complete=True,
        approved_for_cpu_precision_test=False,B1_test_only=True,production_permission_granted=False,
        execution_commit=manifest['code']['commit'],settings=dict(host='Main',backend='cpu',precisions=['float64','float32']),
        original_native_guard_code_objects_preserved=False,original_all_39_plan_guards_preserved=False,
        input_bindings_by_run=deepcopy(blueprint['input_bindings_by_run']),
        source_bindings=[dict(current_source=io.asset(path),original_source=frozen)
            for path,frozen in blueprint['source_provenance'].items()],executed_sources=deepcopy(request['sources']),
        original_metadata_preparation=request['metadata_preparation'],prior_verified_stage=request['prior_verified_stage'],
        actual_owned_copy_receipts=deepcopy(request['copy_receipts']),shared_metadata_receipt=request['shared_metadata_receipt'],
        original_logical_closure=blueprint['original_logical_closure'],preexisting_logical_aliases=deepcopy(blueprint['logical_aliases']),
        numerical_equivalence_verified=False,scientific_arrays_read=False,actual_Root_decision_pending=True)


def stage_from_evidence(manifest,manifest_bound,*,prior_receipt,copy_receipts,closure_bound,preparation_bound,
                        metadata_locals,cache_root,stream,io,expected_io_source,shared_metadata_receipt=None):
    """Adopt exact old SHA-once pins and actual verified-copy pins; stat only.

    The producer receipts remain explicit parents. No existing scientific bytes
    are rehashed or silently requalified. Newly copied small metadata is already
    verified by NasIO.copy_file and receives its own current local pin.
    """
    old=io.read_json(prior_receipt,max_bytes=prior_receipt['size_bytes'])
    if (old.get('format')!='darsia-production-verified-stage-v1' or old.get('complete') is not True
            or old.get('payload_sha_verified_by_coordinator') is not True or old.get('arrays_decoded') is not False
            or set(copy_receipts)!=set(COUNTS)):
        raise ValueError('Exact complete original-stage and allthree owned-copy evidence required')
    known={};requested={stream._key(v['path']) for v in manifest['inputs'].values()}
    for run,copy_receipt in copy_receipts.items():
        copy=io.read_json(copy_receipt,max_bytes=copy_receipt['size_bytes'])
        if (copy.get('format')!='darsia-production-owned-input-stage-v1' or copy.get('complete') is not True
                or copy.get('passed') is not True or copy.get('arrays_decoded') is not False
                or copy.get('scientific_computation_started') is not False
                or not descriptor_equal(copy['closure'],closure_bound,stream._key)
                or not descriptor_equal(copy['metadata_preparation'],preparation_bound,stream._key)
                or not descriptor_equal(copy['io_source'],expected_io_source,stream._key)
                or copy.get('completed_count')!=copy.get('expected_count') or copy.get('failure') is not None
                or copy.get('working_runs')!=[run] or len(copy['copied_assets'])!=copy['completed_count']):
            raise ValueError('Exact complete owned-copy source/count/scope evidence required')
        for row in copy['copied_assets']:
            logical,local=_bound(row['logical_descriptor']),_bound(row['local_descriptor'])
            if (logical['sha256'],logical['size_bytes'])!=(local['sha256'],local['size_bytes']):
                raise ValueError('Copied local input binding differs')
            actual=stream.local_identity(local['path']);old_stamp=row['actual_local_file_identity']
            if any(actual[a]!=old_stamp[b] for a,b in [('size_bytes','size_bytes'),('mtime_ns','mtime_ns'),('device','device'),('file_id','inode')]):
                raise ValueError('Owned copied input identity changed since verified publication')
            if row.get('array_decoded') is not False or row.get('working_runs')!=copy['working_runs']:
                raise ValueError('Owned copy producer scope differs')
            key=stream._key(logical['path'])
            if key in known and (known[key]['logical']['sha256'],known[key]['logical']['size_bytes'])!=(logical['sha256'],logical['size_bytes']):
                raise ValueError('Two producer receipts bind contradictory logical bytes')
            known[key]=dict(logical=logical,local=local,local_identity=actual)
    if shared_metadata_receipt is not None:
        shared=io.read_json(shared_metadata_receipt,max_bytes=shared_metadata_receipt['size_bytes'])
        if (shared.get('complete') is not True or shared.get('passed') is not True
                or shared.get('arrays_decoded') is not False or shared.get('scientific_computation_started') is not False
                or not descriptor_equal(shared['closure'],closure_bound,stream._key)
                or not descriptor_equal(shared['metadata_preparation'],preparation_bound,stream._key)
                or not descriptor_equal(shared['io_source'],expected_io_source,stream._key)
                or shared.get('completed_count')!=3 or len(shared['copied_assets'])!=3):
            raise ValueError('Exact actual shared-three metadata copy proof required')
        for row in shared['copied_assets']:
            logical,local=_bound(row['logical_descriptor']),_bound(row['local_descriptor'])
            if Path(logical['path']).suffix.lower() not in {'.json','.md'} or logical['size_bytes']>64*1024**2:
                raise ValueError('Shared receipt may introduce only exact small metadata')
            stamp=stream.local_identity(local['path']);expected=row['actual_local_file_identity']
            if (logical['sha256'],logical['size_bytes'])!=(local['sha256'],local['size_bytes']) or any(
                    stamp[a]!=expected[b] for a,b in [('size_bytes','size_bytes'),('mtime_ns','mtime_ns'),('device','device'),('file_id','inode')]):
                raise ValueError('Shared copied metadata identity differs')
            known[stream._key(logical['path'])]=dict(logical=logical,local=local,local_identity=stamp)
    # Explicit newly verified small metadata overrides missing old local pins.
    for logical,local in metadata_locals:
        logical,local=_bound(logical),_bound(local)
        if (logical['sha256'],logical['size_bytes'])!=(local['sha256'],local['size_bytes']):
            raise ValueError('Fresh metadata copy changed the frozen logical bytes')
        known[stream._key(logical['path'])]=dict(logical=logical,local=local,local_identity=stream.local_identity(local['path']))
    # Ignore deleted old working inputs replaced by the exact new copy receipts.
    # Only requested shared logical rows not superseded are carried from old pins.
    for row in old['inputs'].values():
        key=stream._key(row['logical']['path'])
        if key not in requested or key in known:continue
        if stream.local_identity(row['local']['path'])!=row['local_identity']:
            raise ValueError('Requested shared staged input changed; never rehash/requalify it')
        known[key]=deepcopy(row)
    output,staged={},{}
    for role,logical in manifest['inputs'].items():
        key=stream._key(logical['path'])
        if key not in known:raise ValueError('No actual SHA-once producer exists for '+role)
        row=deepcopy(known[key])
        if (logical['sha256'],logical['size_bytes'])!=(row['logical']['sha256'],row['logical']['size_bytes']):
            raise ValueError('Producer does not bind the requested logical bytes')
        stream._local(row['local']['path'],cache_root)
        row['logical']=logical;output[role]=row;staged[role]=row['local']
    receipt=dict(format='darsia-production-verified-stage-v1',complete=True,manifest=manifest_bound,
        cache_root=str(Path(cache_root).resolve()),inputs=output,payload_sha_verified_by_coordinator=True,
        prior_verified_stage=prior_receipt,owned_verified_copy=copy_receipts,
        shared_metadata_copy=shared_metadata_receipt,
        payload_sha_rechecked_during_adoption=False,arrays_decoded=False,
        subsequent_read_policy='pinned_local_identity_no_rehash',production_permission_granted=False)
    return receipt,staged


def prepare_metadata(request, output_root):
    """Root metadata-only API. Requires ACTUAL stage receipts and source review.

    metadata_blueprint() supplies review input tables before this call. Root's
    approval is an external file; this function cannot create an approved review.
    Only small explicit code/JSON/TOML metadata is copied. Never scientific data.
    """
    baseline=load_baseline(request['sources']);io=baseline['production_io'].NasIO();jobs=baseline['production_job']
    stream=load_source('_precision_metadata_stream',request['sources']['stream_io'])
    blueprint=metadata_blueprint(request,io=io)
    closure=io.read_json(blueprint['closure'],max_bytes=blueprint['closure']['size_bytes'])
    declared={stream._key(row['descriptor']['path']):row for row in closure['assets']}
    additional=request.get('additional_metadata_copies',[])
    for bound in additional:
        bound=_bound(bound);row=declared.get(stream._key(bound['path']))
        if (row is None or set(row['working_runs'])!=set(COUNTS) or Path(bound['path']).suffix.lower()!='.json'
                or bound['size_bytes']>64*1024**2 or not descriptor_equal(bound,row['descriptor'],stream._key)):
            raise ValueError('Additional copies must be exact original shared JSON declarations')
    review=io.read_json(_bound(request['source_review']),max_bytes=request['source_review']['size_bytes'])
    if (review.get('complete') is not True or review.get('approved_for_cpu_precision_test') is not True
            or review.get('input_bindings_by_run')!=blueprint['input_bindings_by_run']
            or review.get('executed_sources')!=request['sources'] or review.get('production_permission_granted') is not False):
        raise ValueError('Root must supply the actual source review for these exact logical127 tables')
    template=io.read_json(request['manifest_template'],max_bytes=request['manifest_template']['size_bytes'])
    if jobs.code_identity(request['worktree'])!=template['code'] or jobs.environment_lock()!=template['environment']:
        raise ValueError('Current clean committed runtime/environment differs')
    root=Path(output_root).absolute()
    if str(root).startswith(('\\\\','//')) or root.exists() or not root.is_relative_to(Path(request['cache_root']).resolve()):
        raise ValueError('Fresh owned local metadata root inside the common staged cache required')
    root.mkdir(parents=True,exist_ok=False)
    metadata_locals=[]
    original_metadata=io.read_json(request['metadata_preparation'],max_bytes=request['metadata_preparation']['size_bytes'])
    for bound in [request['source_review'],*request['sources'].values(),request['boxes'],*additional,
                  *[row['anchor_receipt'] for row in request['runs'].values()],
                  io.read_json(request['metadata_preparation'],max_bytes=request['metadata_preparation']['size_bytes'])['native_configuration']]:
        bound=_bound(bound)
        if Path(bound['path']).suffix.lower() not in {'.py','.json','.md','.toml'} or bound['size_bytes']>64*1024**2:
            raise ValueError('Only explicit small source/JSON metadata may copy during preparation')
        target=root/'metadata'/bound['sha256']/Path(bound['path']).name;target.parent.mkdir(parents=True,exist_ok=True)
        local=io.copy_file(bound,target);metadata_locals.append((bound,local))
    packet={key:deepcopy(request[key]) for key in ('worktree','python','threads','logical_cpus','sources','source_review',
        'claude_status_path','codex_status_path','authorization','boxes','cache_root')}
    packet.update(format='darsia-production-cpu-precision127-packet-v1',B1_test_only=True,production_permission_granted=False,
        host='Main',backend='cpu',worker_count=3,memory_floor_bytes=20*GIB,reserve_bytes=40*GIB,worker_budget_bytes=20*GIB,
        selected_calibration=blueprint['selected_calibration'],selection=blueprint['selection'],closure=blueprint['closure'],
        source_provenance=blueprint['source_provenance'],logical_aliases=blueprint['logical_aliases'],limits=_bound(request['limits']),
        executed_source_assets=[io.asset(path) for path in blueprint['source_provenance']]+list(request['sources'].values()),
        run_order=sorted(COUNTS,key=lambda r:(-COUNTS[r],r)),coordinator_fence=str(root/'coordinator_fence.json'),
        completed_receipt=str(root/'completed.json'),failure_receipt=str(root/'failed.json'),runs={})
    closure=io.read_json(blueprint['closure'],max_bytes=blueprint['closure']['size_bytes'])
    packet['opaque_descriptors']=[row['descriptor'] for row in closure['assets'] if row['descriptor']['sha256']=='4491b8ea239853d441ece789a371e92f6516874b6ea62308eeafb057981d938c']
    reader_preflights={}
    for run in packet['run_order']:
        options=request['runs'][run];folder=root/run;folder.mkdir()
        row=dict(host='Main',worker=options['worker'],case_ids=blueprint['case_ids_by_run'][run],
            anchor_receipt=options['anchor_receipt'],native_ready=str(folder/'native_ready.json'),release=str(folder/'release.json'),
            worker_log=str(folder/'worker.log'),completed_receipt=str(folder/'completed.json'),owned_input_stage=request['copy_receipts'][run])
        for precision in ('float64','float32'):
            inputs=deepcopy(blueprint['input_bindings_by_run'][run]);science=list(inputs)
            assurance={'SCI_source_review':request['source_review'],**{'execution_'+k:v for k,v in request['sources'].items()}}
            inputs.update(assurance)
            manifest=dict(format='darsia-production-job-v1',job_id=uuid.uuid4().hex,created_utc=datetime.now(timezone.utc).isoformat(),
                purpose='equivalence',code=deepcopy(template['code']),environment=deepcopy(template['environment']),
                settings=dict(template['settings'],host='Main',backend='cpu',precision=precision,threads=request['threads'],workers=3,logical_cpus=request['logical_cpus']),
                calibration=blueprint['selected_calibration']['asset'],specs=dict(template['specs'],equivalence_limits=request['limits']),
                inputs=inputs,scientific_input_roles=science,assurance_input_roles=list(assurance),frames=row['case_ids'],
                production_permission_granted=False,calibration_test_choice_only=True,
                scientific_extension=dict(source_review=request['source_review'],original_four_proof_claimed_for_127=False))
            jobs.validate_manifest(manifest);manifest_path=folder/(precision+'_manifest.json');io.write_json(manifest_path,manifest);bound=io.asset(manifest_path)
            stage,staged=stage_from_evidence(manifest,bound,prior_receipt=request['prior_verified_stage'],copy_receipts=request['copy_receipts'],
                closure_bound=blueprint['closure'],preparation_bound=request['metadata_preparation'],metadata_locals=metadata_locals,
                cache_root=request['cache_root'],stream=stream,io=io,expected_io_source=request['sources']['production_io'],
                shared_metadata_receipt=request['shared_metadata_receipt'])
            stage_path=folder/(precision+'_verified_stage.json');io.write_json(stage_path,stage)
            validated_stage=stream.VerifiedStage(io.asset(stage_path),manifest,staged,cache_root=request['cache_root'],io=io)
            row[precision]=dict(manifest=bound,verified_stage=io.asset(stage_path),staged_inputs=staged,
                npy_contracts=stream.npy_contracts(validated_stage),cache_root=request['cache_root'],
                configuration=next(v for k,v in inputs.items() if k=='native_configuration'),
                local_root=str(folder/precision),nas_root=str(Path(options['nas_root'])/precision))
            # Real unchanged full constructor/source qualification, metadata
            # only. No native module, setup, frame loader or array API is called.
            cls=extension_reader_class(baseline['production_inputs'].ManifestAssetReader,stream,
                source_review=request['source_review'],executed_sources=request['sources'],run=run,io=io)
            actual_reader=cls(manifest,staged,cache_root=request['cache_root'],worktree=request['worktree'],
                source_provenance=packet['source_provenance'],source_qualification={'source_review':request['source_review']},
                logical_aliases=packet['logical_aliases'],opaque_descriptors=packet['opaque_descriptors'],
                verified_staging_receipt=io.asset(stage_path),array_contracts=row[precision]['npy_contracts'],io=io)
            if (actual_reader.source_qualification_state!='source_reviewed_CPU_precision127_test_only'
                    or actual_reader.logical_aliases!=packet['logical_aliases']):
                raise ValueError('Actual full-table reader/source/alias metadata preflight differs')
            reader_preflights[run+'_'+precision]=dict(complete=True,manifest=bound,alias_count=len(actual_reader.logical_aliases),
                flat_input_roles=len(manifest['inputs']),source_qualification_state=actual_reader.source_qualification_state,
                native_setup_called=False,scientific_arrays_read=False,production_permission_granted=False)
            del actual_reader
        packet['runs'][run]=row
    selection=io.read_json(blueprint['selection'],max_bytes=blueprint['selection']['size_bytes'])
    validate_metadata_packet(packet,selection=selection,closure=closure)
    target=root/'packet.json';io.write_json(target,packet)
    return dict(complete=True,packet=io.asset(target),actual_reader_metadata_preflights=reader_preflights,scientific_arrays_read=False,
        payload_sha_rechecked=False,scientific_process_launched=False,production_permission_granted=False)


def json_scalars(value):
    """Native scalar conversion only; arrays cannot silently become JSON lists."""
    if isinstance(value,Mapping):return {str(key):json_scalars(item) for key,item in value.items()}
    if isinstance(value,(tuple,list)):return [json_scalars(item) for item in value]
    if value is None or type(value) in (str,bool,int,float):return value
    if getattr(value,'shape',None)==() and callable(getattr(value,'item',None)):
        return json_scalars(value.item())
    raise TypeError('Summary may contain native scalar values, never a scientific array')


def compare_frame(packet,run,case_id,result,products,frame,reference_record,*,io,comparator,normalize):
    """One bounded actual comparison; no alternate method or recomputation."""
    import numpy as np
    folder=Path(reference_record['local_directory'])
    io.verify_directory(folder,reference_record['files'])
    reference=io.read_json(folder/'result.json')['result']
    manifests={p:io.read_json(packet['runs'][run][p]['manifest']) for p in ('float64','float32')}
    bindings={p:dict(input_bindings={role:m['inputs'][role] for role in m['scientific_input_roles']},
        frozen_method=packet['source_provenance'],calibration=m['calibration']) for p,m in manifests.items()}
    mapped={}
    try:
        for name in (*comparator.FLOAT_PRODUCTS,'phase_labels'):
            mapped[name]=np.load(folder/(name+'.npy'),mmap_mode='r',allow_pickle=False)
        area=np.load(folder/'pixel_area_cm2.npy',mmap_mode='r',allow_pickle=False)
        # Geometry remains unchanged; different per-pixel area is a scope failure.
        actual=np.asarray(frame.pixel_area_m2,np.float64)*1e4
        if area.shape!=frame.domain.shape or not np.array_equal(area,np.broadcast_to(actual,area.shape)):
            raise ValueError('Candidate physical pixel area differs from original native reference')
        output=comparator.compare_retained(reference,json_scalars({k:v for k,v in result.items() if k not in {'fields','pixel_backend_facts'}}),
            mapped,products,reference_settings=dict(host='Main',backend=reference['pixel_backend'],precision='float64'),
            candidate_settings=dict(host='Main',backend=result['pixel_backend'],precision='float32'),
            reference_bindings=bindings['float64'],candidate_bindings=bindings['float32'],pixel_area_cm2=area,
            kind='cpu_float32_vs_float64',reference_quantities={k:reference[k] for k in ('dilution_factor','affected_pore_volume_m3')},
            candidate_quantities={k:json_scalars(result[k]) for k in ('dilution_factor','affected_pore_volume_m3')},
            normalize_locator=normalize)
        output.update(case_id=case_id,reference_commit_files=reference_record['files'])
        return output
    finally:
        for value in [*mapped.values(),locals().get('area')]:
            if getattr(value,'_mmap',None) is not None:value._mmap.close()


def run_variant(packet, run, precision, *, baseline, extensions, identity, reference=None):
    """Called by released actual worker only; numerical setup begins here."""
    import numpy as np
    import psutil,cv2,threadpoolctl
    sys.path.insert(0,str(Path(packet['worktree'])/'src'))
    from darsia.fluidflower import production_native as native,production_retained as retained
    from darsia.presets.workflows.config.roi_registry import RoiRegistry
    modules=native._modules()  # Original imports may reset thread ENV: restore AFTER preload.
    for key in THREAD_ENV:os.environ[key]=str(packet['threads'])
    cv2.setNumThreads(packet['threads'])
    io=baseline['production_io'].NasIO();jobs=baseline['production_job']
    for name,module in [('production_native',native),('production_retained',retained)]:
        if baseline['production_io'].normalize_locator(module.__file__)!=baseline['production_io'].normalize_locator(packet['sources'][name]['path']):
            raise ValueError('Actual loaded native/retained source path differs')
        io.verify(packet['sources'][name])
    row=packet['runs'][run][precision];manifest=io.read_json(row['manifest'])
    job=jobs.LocalProductionJob(row['manifest'],row['local_root'],row['nas_root'],io=io)
    job.assert_runtime(packet['worktree'])
    if (manifest['settings']['precision']!=precision or manifest['settings']['backend']!='cpu'
            or manifest['settings']['threads']!=packet['threads'] or manifest['settings']['host']!='Main'
            or manifest['purpose']!='equivalence' or manifest['calibration']['sha256']!=B1_SHA):
        raise ValueError('Actual immutable variant manifest/settings/B1 differ')
    cls=extension_reader_class(baseline['production_inputs'].ManifestAssetReader,extensions['stream_io'],
        source_review=packet['source_review'],executed_sources=packet['sources'],run=run,io=io)
    reader=cls(manifest,row['staged_inputs'],cache_root=row['cache_root'],worktree=packet['worktree'],
        source_provenance=packet['source_provenance'],source_qualification={'source_review':packet['source_review']},
        logical_aliases=packet['logical_aliases'],opaque_descriptors=packet['opaque_descriptors'],
        verified_staging_receipt=row['verified_stage'],array_contracts=row['npy_contracts'],io=io)
    configuration=reader.json(row['configuration'])
    predicate=candidate_gate(native,manifest,packet['source_review'],run)
    native_view=_Proxy(native,prepare_native_context=_clone(native.prepare_native_context,_candidate_reader=predicate),_modules=lambda:modules)
    def boundary():
        guard(packet,io,identity,job_module=jobs,psutil=psutil,threads=packet['threads'],cv2=cv2,poolctl=threadpoolctl)
    with threadpoolctl.threadpool_limits(limits=packet['threads']):
        boundary();facts=thread_facts(packet['threads'],cv2=cv2,poolctl=threadpoolctl)
        # This original setup reads physical coordinate/floor assets only AFTER release.
        with reader.frame_inputs(boundary_guard=boundary):
            context=extensions['selection'].prepare_precision_context(configuration,reader,native_module=native_view,boundary_guard=boundary)
        if len(context.cases)!=127:raise ValueError('Actual full127 context admission missing')
        cases=packet['runs'][run]['case_ids']
        context.cases={case:context.cases[case] for case in cases}
        anchor=_anchor(reader,packet['runs'][run]['anchor_receipt'],context,run)
        rois=RoiRegistry().load(reader.path(packet['boxes'])).resolve_rois(['box1','box2'])
        adapter=extensions['adapter'].PrecisionAdapter(context,precision,
            pixel_math=extensions['kernel'].PixelMath(precision),retained_module=retained)
        prefetch=extensions['lifecycle'].OneFramePrefetch(reader,max_bytes=GIB)
        boundary();setup_facts=thread_facts(packet['threads'],cv2=cv2,poolctl=threadpoolctl)
        records=[];ref_rows={} if reference is None else {v['case_id']:v for v in reference['cases']}
        try:
            prefetch.submit(cases[0],_prefetch_plan(context,reader,cases[0]))
            for index,case_id in enumerate(cases):
                boundary()
                frame_id=case_id
                if job._frame_path(frame_id).exists():
                    raise FileExistsError('Fresh benchmark never overwrites/reinterprets an existing final')
                plan=_prefetch_plan(context,reader,case_id)
                with prefetch.consume(case_id,boundary_guard=boundary) as prefetched:
                    reader._prefetched={reader._canonical_key(reader._stage_logical(plan[name])['path']):value for name,value in prefetched.items()}
                    if index+1<len(cases):prefetch.submit(cases[index+1],_prefetch_plan(context,reader,cases[index+1]))
                    with reader.frame_inputs(boundary_guard=boundary):
                        wall,cpu=time.perf_counter_ns(),time.process_time_ns()
                        loaded=adapter.load_response(case_id,selected_calibration=packet['selected_calibration'])
                        response=loaded['response'];frame=response['frame'];d=context.modules['engine'].r1.darsia
                        image=response['analysis']._scalar(frame.phase_labels,frame.elapsed_h)
                        masks={'full_rig':frame.domain,**{name:d.roi_to_mask(roi.roi,image,mode='voxels').img for name,roi in rois.items()}}
                        result=adapter.evaluate_main(response,anchor,selected_calibration=packet['selected_calibration'],domain_masks=masks,kept_fields=retained.FIELDS)
                        export=adapter.main_products(result)
                        elapsed=dict(wall_ns=time.perf_counter_ns()-wall,cpu_ns=time.process_time_ns()-cpu)
                        boundary()
                        comparison=None
                        if precision=='float32':
                            comparison=compare_frame(packet,run,case_id,result,export['products'],frame,
                                ref_rows[case_id],io=io,comparator=extensions['comparator'],normalize=baseline['production_io'].normalize_locator)
                        temporary=job.begin_frame(frame_id)
                        # Separate uncompressed NPY products support bounded mmap comparison.
                        for name,array in export['products'].items():np.save(temporary/(name+'.npy'),array,allow_pickle=False)
                        np.save(temporary/'pixel_area_cm2.npy',np.broadcast_to(np.asarray(frame.pixel_area_m2,np.float64),frame.domain.shape)*1e4,allow_pickle=False)
                        metadata=json_scalars({key:value for key,value in result.items() if key not in {'fields','pixel_backend_facts'}})
                        record=dict(case_id=case_id,result=metadata,timing=elapsed,thread_facts=thread_facts(packet['threads'],cv2=cv2,poolctl=threadpoolctl),
                            actual_worker=identity,manifest=row['manifest'],source_review=packet['source_review'],sources=packet['sources'],
                            anchor=packet['runs'][run]['anchor_receipt'],comparison=comparison,
                            B1_test_only=True,production_permission_granted=False)
                        io.write_json(temporary/'result.json',record)
                        job.commit_frame(frame_id,temporary,product_metadata={'precision':precision,'backend':'cpu','units':export['units'],
                            'working_pixel_dtype':precision,'source_review':packet['source_review'],'scientific_eligibility':False,'production_permission_granted':False})
                        published=job.copy_frame_to_nas(frame_id)
                        records.append(dict(case_id=case_id,nas_directory=published['nas_directory'],files=published['files'],local_directory=str(job._frame_path(frame_id)),
                            comparison=comparison))
                        reader._prefetched.clear()
                        del image,masks,export,result,response,frame,loaded;context.captured.clear();gc.collect()
                        if comparison is not None and comparison['limits_passed'] is not True:
                            raise RuntimeError('Actual add8v2 precision comparison failed; both products retained')
        finally:
            reader._prefetched={}
            prefetch.close()
        boundary()
    # Numerical caches are released before switching precision; source objects
    # and immutable shared staged files remain unchanged.
    del adapter,context,reader;gc.collect()
    return dict(precision=precision,cases=records,initial_thread_facts=facts,setup_thread_facts=setup_facts,actual_worker=identity,
                whole_run=run,chronological=True,production_permission_granted=False)


def worker(packet_descriptor, run):
    """Explicit Root entrypoint. Native PID/epoch handshake precedes NumPy/DarSIA."""
    # Packet itself is metadata and bounded before baseline source loading.
    raw=Path(_bound(packet_descriptor)['path']).read_bytes()
    if len(raw)!=packet_descriptor['size_bytes'] or hashlib.sha256(raw).hexdigest()!=packet_descriptor['sha256']:
        raise ValueError('Worker packet SHA/size differs')
    packet=json.loads(raw);packet['packet_sha256']=packet_descriptor['sha256']
    baseline=load_baseline(packet['sources']);io=baseline['production_io'].NasIO();jobs=baseline['production_job']
    import psutil
    identity=jobs.worker_identity();row=packet['runs'][run]
    for key in THREAD_ENV:os.environ[key]=str(packet['threads'])
    guard(packet,io,identity,job_module=jobs,psutil=psutil)
    if 'numpy' in sys.modules or 'darsia' in sys.modules:
        raise RuntimeError('Actual native handshake must precede NumPy/DarSIA imports')
    io.write_json(row['native_ready'],dict(complete=True,actual_native_worker=identity,packet=packet_descriptor,run=run,
        sources=packet['sources'],scientific_arrays_read=False,numpy_or_darsia_imported=False,production_permission_granted=False))
    deadline=time.monotonic()+packet.get('handshake_timeout_seconds',120)
    while not Path(row['release']).exists():
        guard(packet,io,identity,job_module=jobs,psutil=psutil)
        if time.monotonic()>=deadline:raise TimeoutError('Root native handshake release absent')
        time.sleep(.05)
    release=io.read_json(row['release'])
    if release.get('packet')!=packet_descriptor or release.get('workers',{}).get(run)!=identity:
        raise ValueError('Root release does not bind actual native PID/start time')
    extensions={}
    for name in ('stream_io','lifecycle','kernel','adapter','selection','comparator'):
        module_name='production_precision' if name=='kernel' else '_benchmark_'+name
        extensions[name]=load_source(module_name,packet['sources'][name])
    reference=run_variant(packet,run,'float64',baseline=baseline,extensions=extensions,identity=identity)
    candidate=run_variant(packet,run,'float32',baseline=baseline,extensions=extensions,identity=identity,reference=reference)
    result=dict(complete=True,run=run,actual_worker=identity,reference=reference,candidate=candidate,
                all_frame_limits_passed=all(v['comparison']['limits_passed'] for v in candidate['cases']),
                Wasserstein_performed=False,whole_production_qualification_complete=False,
                source_review=packet['source_review'],production_permission_granted=False)
    io.write_json(row['completed_receipt'],result)
    return result


def verify_packet_metadata(packet_bound, *, io):
    packet=io.read_json(_bound(packet_bound),max_bytes=packet_bound['size_bytes'])
    selection=io.read_json(packet['selection'],max_bytes=packet['selection']['size_bytes'])
    closure=io.read_json(packet['closure'],max_bytes=packet['closure']['size_bytes'])
    validate_metadata_packet(packet,selection=selection,closure=closure)
    for source in packet['executed_source_assets']:io.verify(source)
    review=io.read_json(packet['source_review'],max_bytes=packet['source_review']['size_bytes'])
    if (review.get('complete') is not True or review.get('approved_for_cpu_precision_test') is not True
            or review.get('production_permission_granted') is not False
            or review.get('executed_sources')!=packet['sources']):
        raise ValueError('Actual external Root source review is missing or names different code')
    packet['packet_sha256']=packet_bound['sha256']
    return packet


def qualify_native_ready(value,*,child,wrapper_identity,argv,packet_bound,run,packet,jobs,psutil,normalize):
    """Native incarnation/argv + exact wrapper ancestry, never a Popen PID guess."""
    actual=value.get('actual_native_worker',{})
    if (value.get('complete') is not True or value.get('packet')!=packet_bound or value.get('run')!=run
            or value.get('sources')!=packet['sources'] or value.get('scientific_arrays_read') is not False
            or value.get('numpy_or_darsia_imported') is not False or type(actual.get('pid')) is not int
            or not isinstance(actual.get('process_create_time'),str)):
        raise ValueError('Actual native source/metadata-first handshake differs')
    if jobs.worker_identity(child.pid)!=wrapper_identity or jobs.worker_identity(actual['pid'])!=actual:
        raise ValueError('Actual wrapper/native PID incarnation changed')
    process=psutil.Process(actual['pid'])
    if actual['pid']==child.pid or not any(p.pid==child.pid for p in process.parents()[:3]):
        raise ValueError('Claimed native is not the launched wrapper descendant')
    observed=process.cmdline()
    if len(observed)!=len(argv):raise ValueError('Actual native command argument count differs')
    # argv0 may be the redirector's native interpreter. The fixed script/packet
    # arguments bind the actual code path and exact whole-run invocation.
    paths={argv.index('worker')-1,argv.index('--packet')+1}
    if any((normalize(observed[i])!=normalize(argv[i]) if i in paths else observed[i]!=argv[i])
           for i in range(1,len(argv))):
        raise ValueError('Actual native argv does not bind this runner/packet/run')
    return actual


def coordinate(packet_bound, *, before_launch, run_stream=None, user_deletion_go=None):
    """Root explicitly calls this; no flag in a manifest can authorize launch.

    Inputs are already staged by the Root-owned bounded host copier. A supplied
    HostRunStream must own those exact directories/assignment before launch.
    This function never starts another experiment or infers production GO.
    """
    if not callable(before_launch):raise PermissionError('Fresh trusted Root launch guard required')
    import subprocess,psutil
    raw=Path(_bound(packet_bound)['path']).read_bytes()
    if len(raw)!=packet_bound['size_bytes'] or hashlib.sha256(raw).hexdigest()!=packet_bound['sha256']:
        raise ValueError('Actual immutable packet differs')
    basic=json.loads(raw);baseline=load_baseline(basic['sources']);io=baseline['production_io'].NasIO();jobs=baseline['production_job']
    packet=verify_packet_metadata(packet_bound,io=io);owner=jobs.worker_identity()
    if psutil.cpu_count()!=packet['logical_cpus'] or psutil.virtual_memory().available<=100*GIB:
        raise RuntimeError('Actual Main CPU/three20GiB workers plus40GiB reserve admission failed')
    io.verify(packet['python'])
    if run_stream is not None:
        if (run_stream.host!='Main' or {r:w for r,w in run_stream._allowed.items()}!=
                {r:row['worker'] for r,row in packet['runs'].items()}):
            raise ValueError('Actual host lifecycle differs from the immutable whole-run assignment')
    processes={};ready={};handles=[];wrappers={};commands={}
    def current(continue_flag=True):
        guard(packet,io,owner,job_module=jobs,psutil=psutil,check_coordinator=False)
        return dict(owner=owner,worker_packet_sha256=packet_bound['sha256'],continue_=continue_flag,
            updated_utc=datetime.now(timezone.utc).isoformat())
    def publish_fence(continue_flag=True):
        def provider():
            value=current(continue_flag);value['continue']=value.pop('continue_');return value
        if packet.get('coordinator_fence'):
            with local_fence_mutex(packet_bound['sha256']):io.replace_json(packet['coordinator_fence'],provider)
    try:
        # Declared longest-first order is immutable. One complete AC per worker.
        for run in packet['run_order']:
            before_launch();publish_fence();row=packet['runs'][run]
            if run_stream is not None and run_stream.workers[row['worker']].active!=run:
                raise ValueError('Whole-run lifecycle must be active before the child starts')
            if psutil.virtual_memory().available<=40*GIB+20*GIB*(3-len(processes)):
                raise RuntimeError('Unlaunched worker reservations would consume the preserved reserve')
            log=Path(row['worker_log']);log.parent.mkdir(parents=True,exist_ok=True)
            handle=log.open('xb');handles.append(handle)
            argv=[packet['python']['path'],'-I','-B',packet['sources']['runner']['path'],'worker',
                '--packet',packet_bound['path'],'--sha256',packet_bound['sha256'],'--size',str(packet_bound['size_bytes']),'--run',run]
            processes[run]=subprocess.Popen(argv,stdout=handle,stderr=subprocess.STDOUT)
            wrappers[run]=jobs.worker_identity(processes[run].pid);commands[run]=argv
        deadline=time.monotonic()+packet.get('handshake_timeout_seconds',120)
        while len(ready)!=3:
            publish_fence()
            for run,child in processes.items():
                if child.poll() is not None:raise RuntimeError('Worker exited before actual native PID handshake')
                path=Path(packet['runs'][run]['native_ready'])
                if run not in ready and path.exists():
                    value=io.read_json(path)
                    actual=qualify_native_ready(value,child=child,wrapper_identity=wrappers[run],argv=commands[run],
                        packet_bound=packet_bound,run=run,packet=packet,jobs=jobs,psutil=psutil,
                        normalize=baseline['production_io'].normalize_locator)
                    ready[run]=actual
            if time.monotonic()>deadline:raise TimeoutError('All three actual native handshakes did not arrive')
            time.sleep(.1)
        for run,identity in ready.items():
            for precision in ('float64','float32'):
                row=packet['runs'][run][precision]
                job=jobs.LocalProductionJob(row['manifest'],row['local_root'],row['nas_root'],io=io)
                job.register_worker(identity,frame_id=packet['runs'][run]['case_ids'][0])
            io.write_json(packet['runs'][run]['release'],dict(packet=packet_bound,workers={run:identity},
                B1_test_only=True,production_permission_granted=False))
        while any(child.poll() is None for child in processes.values()):
            publish_fence()
            if any(child.poll() not in (None,0) for child in processes.values()):
                raise RuntimeError('One worker failed; local freshness fence stops peers without kills/retries')
            time.sleep(1)
        if any(child.returncode!=0 for child in processes.values()):raise RuntimeError('Actual worker exit failure')
        all_identities=[*ready.values(),*wrappers.values()]
        jobs.require_workers_retired(all_identities)
        receipts={run:io.read_json(packet['runs'][run]['completed_receipt']) for run in ready}
        for run,result in receipts.items():
            if (result.get('complete') is not True or result.get('actual_worker')!=ready[run]
                    or result.get('all_frame_limits_passed') is not True
                    or [v['case_id'] for v in result['reference']['cases']]!=packet['runs'][run]['case_ids']
                    or [v['case_id'] for v in result['candidate']['cases']]!=packet['runs'][run]['case_ids']):
                raise ValueError('Actual completed127 history/worker comparison receipt differs')
        if run_stream is not None:
            for run in packet['run_order']:
                run_stream.finish(packet['runs'][run]['worker'],run,
                    workers_retired_guard=lambda:jobs.require_workers_retired(all_identities))
        summary=dict(complete=True,all_127_frame_limits_passed=True,workers=ready,wrappers=wrappers,results=receipts,
            source_review=packet['source_review'],Wasserstein_performed=False,production_qualification_complete=False,
            production_permission_granted=False,cleanup_pending=True,cleanup_requested=user_deletion_go is not None)
        io.write_json(packet['completed_receipt'],summary)
        if user_deletion_go is not None:
            cleanup_completed(packet,receipts,io=io,retired=lambda:jobs.require_workers_retired(all_identities),
                user_deletion_go=user_deletion_go)
            io.write_json(Path(packet['completed_receipt']).with_name('cleanup_completed.json'),dict(complete=True,
                input_cleanup_remains_HostRunStream_owned=True,output_cleanup_performed=True,
                actual_retired_workers=all_identities,production_permission_granted=False))
        return summary
    except BaseException as error:
        # No process is killed/relaunched. Fence naturally expires if a current
        # guard denies publication; original scientific outputs/partials remain.
        try:publish_fence(False)
        except BaseException:pass
        io.write_json(packet['failure_receipt'],dict(complete=False,error_type=type(error).__name__,error=str(error),
            workers=ready,wrappers=wrappers,launched_pids={r:p.pid for r,p in processes.items()},
            preserved_all_finals_and_partials=True,production_permission_granted=False))
        raise
    finally:
        for handle in handles:handle.close()


def cleanup_completed(packet,receipts,*,io,retired,user_deletion_go):
    """Only closed fresh output files; local inputs remain with HostRunStream.

    The separate Root lifecycle cleanup owns each run's input directories and
    its exact original NAS counterparts. Common cached calibrations are outside
    this function. No partial, old final or operation receipt is traversed.
    """
    if not callable(user_deletion_go) or not callable(retired):raise PermissionError('Trusted session cleanup/retirement required')
    plans=[];retired()
    for run,result in receipts.items():
        for precision in ('reference','candidate'):
            for row in result[precision]['cases']:
                folder=Path(row['local_directory']);files=row['files']
                root=Path(packet['runs'][run]['float64' if precision=='reference' else 'float32']['local_root'])/'frames'
                if folder.resolve().parent!=root.resolve() or folder.name!=row['case_id'] or folder.name.startswith('.'):
                    raise ValueError('Only fresh manifest-owned committed frame files can be cleaned')
                io.verify_directory(row['nas_directory'],files);io.verify_directory(folder,files)
                paths=[folder/name for name in files]
                if any(p.is_symlink() or p.name.endswith('.partial') for p in paths):raise ValueError('Partials are always preserved')
                stamps={p:(p.stat().st_dev,p.stat().st_ino,p.stat().st_size,p.stat().st_mtime_ns,p.stat().st_ctime_ns) for p in paths}
                plans.append((run,folder,paths,stamps))
    owned=tuple(str(folder) for _,folder,_,_ in plans)
    if user_deletion_go(owned) is not True:raise PermissionError('Explicit current user cleanup authorization absent')
    retired()
    for _,_,paths,stamps in plans:
        for p in paths:
            if (p.stat().st_dev,p.stat().st_ino,p.stat().st_size,p.stat().st_mtime_ns,p.stat().st_ctime_ns)!=stamps[p]:
                raise ValueError('Owned local result changed after exact NAS verification')
    if user_deletion_go(owned) is not True:raise PermissionError('Current cleanup authorization changed')
    for _,folder,paths,_ in plans:
        for path in paths:path.unlink()
        folder.rmdir()


def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['worker']);parser.add_argument('--packet',required=True)
    parser.add_argument('--sha256',required=True);parser.add_argument('--size',type=int,required=True)
    parser.add_argument('--run',choices=sorted(COUNTS),required=True)
    args=parser.parse_args(argv)
    bound=dict(path=str(Path(args.packet).absolute()),sha256=args.sha256,size_bytes=args.size)
    try:worker(bound,args.run)
    except BaseException as error:
        print(json.dumps(dict(error_type=type(error).__name__,error=str(error))),file=sys.stderr,flush=True);raise


if __name__=='__main__':main()
