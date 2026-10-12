"""Per-worker whole-run ownership and one-frame IO-only prefetch; no science GO.

The existing coordinator owns guards, workers and verified publication. This
module owns ONLY directories it freshly creates. Old cache/finals/partials and
common frozen calibration/source files are never adopted into that ownership.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, ExitStack
from copy import deepcopy
import math
import os
from pathlib import Path
import re
import threading
from collections.abc import Mapping


def classify_inputs(closure, per_run_bindings, *, normalize):
    """Explicit closure membership, never infer scientific scope from filenames.

    Every unassigned or multi-run asset remains common. Original global plan,
    calibration/model/freeze/code guards override a proposed run assignment.
    The caller supplies actual per-run dependencies from the locked metadata.
    """
    if closure.get('format') != 'darsia-production-native-flat-closure-v1' or closure.get('complete') is not True:
        raise ValueError('Original complete flat native closure required')
    rows = {normalize(item['descriptor']['path']):item for item in closure['assets']}
    if len(rows) != len(closure['assets']):
        raise ValueError('Duplicate flat closure locator')
    assignments = {}
    for run, descriptors in per_run_bindings.items():
        if re.fullmatch(r'ac[0-9]{2}', run) is None:
            raise ValueError('Explicit original AC run ID required')
        for bound in descriptors:
            key = normalize(bound['path'])
            if key not in rows or any(rows[key]['descriptor'].get(k) != bound.get(k) for k in ('sha256','size_bytes')):
                raise ValueError('Run dependency differs from the original flat closure')
            assignments.setdefault(key, set()).add(run)
    result = dict(common=[], runs={run:[] for run in per_run_bindings})
    protected = ('original_plan_', 'actual_B1', 'original_R1_', 'original_campaign_code',
                 'campaign', 'locked_selection', 'R2_input_plan', 'base_input_plan',
                 'recipes', 'atmosphere', 'dose_inventory', 'light_epochs', 'original_bottom_membership')
    for key, item in rows.items():
        runs = assignments.get(key, set())
        if len(runs) != 1 or any(role.startswith(protected) for role in item['roles']):
            result['common'].append(deepcopy(item['descriptor']))
        else:
            result['runs'][next(iter(runs))].append(deepcopy(item['descriptor']))
    return result


class OneFramePrefetch:
    """One next frame; read-only copies of fixed NPY pixels/validity/labels.

    No reduction, model, image decode, scientific function or authority call is
    run in the I/O thread. Fresh original guards are supplied at consumption.
    """
    def __init__(self, reader, *, max_bytes=1024**3):
        if type(max_bytes) is not int or not 0 < max_bytes <= 1024**3:
            raise ValueError('One-frame prefetch must stay at or below 1 GiB')
        if getattr(reader, '_verified_stage', None) is None:
            raise ValueError('Prefetch needs the SHA-once pinned stage reader')
        self.reader, self.max_bytes = reader, max_bytes
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='native-input-io')
        self._future = None
        self._frame = None
        self._consuming = False
        self._consuming_frame = None
        self._lock = threading.Lock()

    def submit(self, frame_id, descriptors):
        if not isinstance(frame_id, str) or not frame_id or not descriptors:
            raise ValueError('One explicit next frame and its NPY bindings required')
        if set(descriptors)-{'centre','ring','centre_valid','ring_valid','valid','labels'}:
            raise ValueError('Only next-frame RGB/validity/labels may be prefetched')
        import numpy as np
        plan, count = {}, 0
        for name, request in descriptors.items():
            logical = self.reader._stage_logical(request)
            contract = self.reader._array_contracts.get(self.reader._canonical_key(logical['path']))
            if contract is None or Path(logical['path']).suffix.lower() != '.npy':
                raise ValueError('Prefetch requires the exact staged NPY header contract')
            dtype = np.dtype(contract['dtype'])
            if dtype.hasobject:
                raise ValueError('Object arrays cannot be prefetched')
            if (name == 'labels' and not np.issubdtype(dtype, np.integer)
                    or name in {'valid','centre_valid','ring_valid'} and dtype != np.dtype(bool)
                    or name in {'centre','ring'} and dtype != np.dtype('float32')):
                raise ValueError('Prefetch must preserve original RGB/boolean/categorical storage dtype')
            size = math.prod(contract['shape'])*dtype.itemsize
            count += size
            if count > self.max_bytes or size > logical['size_bytes']:
                raise ValueError('Next-frame prefetch exceeds its exact byte budget')
            plan[name] = (logical, contract)
        with self._lock:
            if self._future is not None:
                raise RuntimeError('Only one next frame may be queued or retained')
            if self._consuming and frame_id == self._consuming_frame:
                raise ValueError('Prefetch must be a different next frame, not the active frame')
            self._frame = frame_id
            self._future = self._executor.submit(self._read, plan)

    def _read(self, plan):
        import numpy as np
        held, arrays = ExitStack(), {}
        try:
            for name, (logical, contract) in plan.items():
                path = held.enter_context(self.reader._verified_stage.hold(logical))
                source = np.load(path, mmap_mode='r', allow_pickle=False)
                try:
                    if tuple(source.shape) != contract['shape'] or source.dtype != np.dtype(contract['dtype']):
                        raise ValueError('Prefetch NPY differs from the exact staged header')
                    arrays[name] = np.array(source, copy=True, order='C')  # Actual bounded IO, no pixel arithmetic.
                    arrays[name].setflags(write=False)
                finally:
                    if getattr(source, '_mmap', None) is not None:
                        source._mmap.close()
                self.reader._verified_stage.check(logical)
            return dict(arrays=arrays, held=held, bindings={name:logical for name,(logical,_) in plan.items()})
        except BaseException:
            held.close()
            raise

    @contextmanager
    def consume(self, frame_id, *, boundary_guard):
        if not callable(boundary_guard):
            raise ValueError('Original fresh frame-boundary guard required')
        with self._lock:
            if self._consuming:
                raise RuntimeError('Only one current frame may consume a prefetch')
            if self._future is None or frame_id != self._frame:
                raise ValueError('Consumer differs from the one queued next frame')
            future = self._future
            # The completed/current frame is now owned by this local context,
            # leaving precisely one pending slot for next-frame IO during work.
            self._future, self._frame = None, None
            self._consuming, self._consuming_frame = True, frame_id
        result = None
        try:
            result = future.result()  # Wait for IO first: gate is fresh immediately before calculation.
            for logical in result['bindings'].values():
                self.reader._verified_stage.check(logical)
            boundary_guard()
            yield result['arrays']
            boundary_guard()
        finally:
            try:
                if result is not None:
                    try:
                        result['held'].close()
                    finally:
                        result['arrays'].clear()
            finally:
                with self._lock:
                    self._consuming, self._consuming_frame = False, None

    def close(self):
        with self._lock:
            if self._consuming:
                raise RuntimeError('Cannot close prefetch while current worker arrays remain in use')
            future = self._future
        if future is not None:
            try:
                value = future.result()
                value['held'].close()
                value['arrays'].clear()
            finally:
                with self._lock:
                    self._future, self._frame = None, None
        self._executor.shutdown(wait=True, cancel_futures=True)


class RunStream:
    """One active AC run, at most one next copying, explicit local ownership.

    Roots must be freshly created. Input/result cleanup is optional and needs
    fresh user authorization plus genuine retirement and exact NAS verification.
    No source/common directory can be registered or deleted by a caller.
    """
    def __init__(self, fresh_root, *, io, stages=(), common_bindings=(), nas_is_network=None,
                 copy_executor=None):
        self.root = Path(fresh_root).absolute()
        if str(self.root).startswith(('\\\\','//')) or self.root.resolve() != self.root:
            raise ValueError('Fresh ordinary local stream root required')
        self.root.mkdir(parents=False, exist_ok=False)
        self.io, self.stages = io, list(stages)
        self._common_contents = {(v['sha256'],v['size_bytes']) for v in common_bindings}
        self._nas_is_network = nas_is_network or (lambda p: str(p).startswith(('\\\\','//')))
        self.runs, self.active, self._next = {}, None, None
        self._lock = threading.RLock()
        self._owns_copy_executor = copy_executor is None
        self._copy_executor = copy_executor or ThreadPoolExecutor(max_workers=1, thread_name_prefix='next-run-copy')

    def _directories(self, run):
        if re.fullmatch(r'ac[0-9]{2}', run) is None:
            raise ValueError('Explicit original AC run ID required')
        if run not in self.runs:
            folder = self.root/run
            folder.mkdir(exist_ok=False)
            for name in ('inputs','results'):
                (folder/name).mkdir(exist_ok=False)
            self.runs[run] = dict(folder=folder, inputs=folder/'inputs', results=folder/'results',
                                 original_inputs={}, original_input_identities={}, input_identities={}, result_inventory=None,
                                 result_identities={}, folder_id=(folder.stat().st_dev,folder.stat().st_ino), state='staged')
        return self.runs[run]

    def _stage(self, run, bindings, guard):
        record = self._directories(run)
        output = {}
        for role, descriptor in bindings.items():
            if (descriptor['sha256'],descriptor['size_bytes']) in self._common_contents:
                raise ValueError('Shared common/calibration inputs cannot enter run deletion ownership')
            path = record['inputs']/descriptor['sha256']/Path(descriptor['path']).name
            path.parent.mkdir(exist_ok=True)
            def source_identity():
                stamp = Path(descriptor['path']).stat()
                if stamp.st_size != descriptor['size_bytes']:
                    raise ValueError('Original frozen source size changed')
                return (stamp.st_dev,stamp.st_ino,stamp.st_size,stamp.st_mtime_ns,stamp.st_ctime_ns)
            before = self.io.retry.run('stat-before-copy', descriptor['path'], source_identity, before_attempt=guard)
            actual = self.io.copy_file(descriptor, path, before_attempt=guard)
            after = self.io.retry.run('stat-after-copy', descriptor['path'], source_identity, before_attempt=guard)
            if before != after:
                raise ValueError('Original source changed across its SHA-verified copy')
            output[role] = actual
            relative = path.relative_to(record['inputs']).as_posix()
            record['original_inputs'][relative] = deepcopy(descriptor)
            record['original_input_identities'][relative] = after
            stamp = path.stat()
            record['input_identities'][relative] = (stamp.st_dev,stamp.st_ino,stamp.st_size,stamp.st_mtime_ns,stamp.st_ctime_ns)
        return output

    def start(self, run, bindings, *, boundary_guard):
        if not callable(boundary_guard):
            raise ValueError('Actual original frame/source guard required')
        with self._lock:
            if self.active is not None:
                raise RuntimeError('Only one AC run may be active')
            if self._next is not None:
                if run != self._next[0]:
                    raise ValueError('Next copying run differs')
                if bindings != self._next[2]:
                    raise ValueError('Next run source bindings differ from its copying request')
                output = self._next[1].result()
                self._next = None
            else:
                output = self._stage(run, bindings, boundary_guard)
            boundary_guard()
            self.active = run
            self.runs[run]['state'] = 'active'
            return output

    def copy_next(self, run, bindings, *, before_copy):
        if not callable(before_copy):
            raise ValueError('Original control guard required at every copy attempt')
        with self._lock:
            if self.active is None or run == self.active or self._next is not None or run in self.runs:
                raise RuntimeError('Exactly one new next run may copy while one run is active')
            copied = deepcopy(bindings)
            self._next = (run, self._copy_executor.submit(self._stage, run, copied, before_copy), copied)

    def finish(self, run, *, workers_retired_guard):
        with self._lock:
            if self.active != run or not callable(workers_retired_guard):
                raise ValueError('Only the actual active run can retire')
            workers_retired_guard()  # Original external exact incarnation/descendant check, no inferred PASS.
            self.runs[run]['state'] = 'retired'
            self.active = None

    def bind_results(self, run, expected_inventory):
        with self._lock:
            record = self.runs[run]
            if record['state'] != 'retired':
                raise ValueError('Results must be closed and actual workers retired')
            self.io.verify_directory(record['results'], expected_inventory)
            record['result_inventory'] = deepcopy(expected_inventory)
            record['result_identities'] = {name:(p.stat().st_dev,p.stat().st_ino,p.stat().st_size,p.stat().st_mtime_ns,p.stat().st_ctime_ns)
                                           for name in expected_inventory for p in [record['results']/name]}

    def cleanup(self, run, *, nas_result_directory, user_deletion_go, workers_retired_guard):
        """No broad rmtree. Verify exact NAS bytes; unlink only exact owned files.

        user_deletion_go(run, owned_paths) must represent fresh explicit user
        authorization, outside immutable manifest data. A manifest flag cannot
        authorize this method. All verification precedes any unlink.
        """
        with self._lock:
            record = self.runs[run]
            if (record['state'] != 'retired' or self.active == run or self._next is not None
                    and self._next[0] == run or record['result_inventory'] is None
                    or not callable(user_deletion_go) or not callable(workers_retired_guard)):
                raise ValueError('Closed owned run with explicit user cleanup authorization required')
            workers_retired_guard()
            owned = (record['inputs'], record['results'])
            stamp = record['folder'].stat()
            if (stamp.st_dev,stamp.st_ino) != record['folder_id']:
                raise ValueError('Owned run directory was replaced')
            if not self._nas_is_network(nas_result_directory) or any(not self._nas_is_network(v['path']) for v in record['original_inputs'].values()):
                raise ValueError('Cleanup requires actual verified NAS counterparts for inputs/results')
            if user_deletion_go(run, tuple(str(p) for p in owned)) is not True:
                raise PermissionError('Fresh user deletion GO absent')
            # Hold stage reference locks through verification/deletion. A caller
            # must keep its worker/frame/prefetch interval open until actual use ends.
            with ExitStack() as locks:
                for stage in self.stages:
                    locks.enter_context(stage._lock)
                    if any(Path(path).is_relative_to(record['folder']) for path in stage.references):
                        raise RuntimeError('Worker/prefetch still references the owned run')
                self.io.verify_directory(nas_result_directory, record['result_inventory'])
                for relative,bound in record['original_inputs'].items():
                    def unchanged_source():
                        stamp = Path(bound['path']).stat()
                        actual = (stamp.st_dev,stamp.st_ino,stamp.st_size,stamp.st_mtime_ns,stamp.st_ctime_ns)
                        if actual != record['original_input_identities'][relative]:
                            raise ValueError('Original NAS source identity changed after its SHA-verified copy')
                    self.io.retry.run('stat-original-before-cleanup', bound['path'], unchanged_source,
                                      before_attempt=workers_retired_guard)
                expected = {'inputs':record['original_inputs'], 'results':record['result_inventory']}
                snapshots = {}
                for kind, folder in (('inputs', record['inputs']), ('results',record['results'])):
                    paths = list(folder.rglob('*'))
                    if (folder.resolve() != folder or any(p.is_symlink() or p.name.endswith('.partial') for p in paths)
                            or {p.relative_to(folder).as_posix() for p in paths if p.is_file()} != set(expected[kind])):
                        raise ValueError('Owned directory inventory changed or contains a preserved partial')
                    for path in paths:
                        if not path.is_file() and not path.is_dir():
                            raise ValueError('Special owned file cannot be deleted')
                        if path.is_file():
                            stamp = path.stat()
                            actual = (stamp.st_dev,stamp.st_ino,stamp.st_size,stamp.st_mtime_ns,stamp.st_ctime_ns)
                            identity = record['input_identities' if kind == 'inputs' else 'result_identities'][path.relative_to(folder).as_posix()]
                            if actual != identity:
                                raise ValueError('Owned local file changed after its verified publication/copy')
                    snapshots[kind] = paths
                # No file is removed until BOTH owned directories and all NAS
                # counterparts passed, and fresh worker retirement was rechecked.
                workers_retired_guard()
                if user_deletion_go(run, tuple(str(p) for p in owned)) is not True:
                    raise PermissionError('Fresh user deletion GO expired')
                for kind, folder in (('inputs',record['inputs']), ('results',record['results'])):
                    for path in snapshots[kind]:
                        if path.is_file():
                            path.unlink()
                    for path in sorted((p for p in snapshots[kind] if p.is_dir()), key=lambda p:len(p.parts), reverse=True):
                        path.rmdir()
                    folder.rmdir()
                record['folder'].rmdir()
                record['state'] = 'removed'

    def close(self):
        if self._owns_copy_executor:
            self._copy_executor.shutdown(wait=True, cancel_futures=True)


def _proof(value):
    if (not isinstance(value, Mapping) or set(value) != {'path','sha256','size_bytes'}
            or not isinstance(value['path'], str) or not os.path.isabs(value['path'])
            or not isinstance(value['sha256'], str) or re.fullmatch('[0-9a-f]{64}', value['sha256']) is None
            or type(value['size_bytes']) is not int or value['size_bytes'] < 0):
        raise ValueError('Actual full SHA/size measurement/equivalence descriptor required')
    return deepcopy(value)


def static_assign_runs(runs, hosts):
    """Longest-first whole-run assignment using explicit actual measured rates.

    total_images_per_hour is the measured HOST rate at exactly measured_workers,
    not a per-worker rate. Caller supplies already verified measurement and
    eligibility evidence. This metadata allocator grants no scientific access.
    """
    if not isinstance(runs, list) or not runs or not isinstance(hosts, list) or not hosts:
        raise ValueError('Actual locked runs and eligible measured hosts required')
    run_ids, prepared = set(), []
    for row in runs:
        if (set(row) != {'run','frame_count'} or re.fullmatch('ac[0-9]{2}', row['run']) is None
                or row['run'] in run_ids or type(row['frame_count']) is not int or row['frame_count'] <= 0):
            raise ValueError('One exact positive locked-frame count per whole AC run required')
        run_ids.add(row['run'])
    slots, host_ids, facts = [], set(), []
    for host in hosts:
        allowed = {'host','workers','measured_workers','total_images_per_hour','measurement',
                   'backend','precision','eligible','equivalence_result','equivalence_passed'}
        if (set(host) != allowed or host['host'] not in {'Main','Olav'} or host['host'] in host_ids
                or type(host['workers']) is not int or host['workers'] <= 0
                or host['measured_workers'] != host['workers'] or type(host['measured_workers']) is not int
                or type(host['total_images_per_hour']) not in (int,float)
                or not math.isfinite(host['total_images_per_hour']) or host['total_images_per_hour'] <= 0
                or host['backend'] not in {'cpu','cuda','opencl'} or host['precision'] not in {'float32','float64'}
                or type(host['eligible']) is not bool or type(host['equivalence_passed']) is not bool):
            raise ValueError('Exact host measurement at its actual worker count required')
        host_ids.add(host['host'])
        _proof(host['measurement'])
        if host['equivalence_result'] is not None:
            _proof(host['equivalence_result'])
        if not host['eligible']:
            continue
        if host['host'] == 'Olav' and (host['equivalence_passed'] is not True or host['equivalence_result'] is None):
            raise ValueError('Olav cannot be assigned before actual add8 equivalence PASS')
        facts.append(deepcopy(host))
        rate = float(host['total_images_per_hour'])/host['workers']
        slots.extend(dict(host=host['host'], worker=i, load_hours=0., rate=rate,
                          backend=host['backend'], precision=host['precision']) for i in range(host['workers']))
    if not slots:
        raise ValueError('No actually eligible measured host')
    if len({slot['precision'] for slot in slots}) != 1:
        raise ValueError('One production set must retain one working precision')
    # No timing adaptive queue: record every whole-run choice before execution.
    for run in sorted(runs, key=lambda r:(-r['frame_count'],r['run'])):
        slot = min(slots, key=lambda s:(s['load_hours']+run['frame_count']/s['rate'],s['host'],s['worker']))
        hours = run['frame_count']/slot['rate']
        prepared.append(dict(run=run['run'],frame_count=run['frame_count'],host=slot['host'],
                             worker=slot['worker'],backend=slot['backend'],precision=slot['precision'],
                             estimated_hours=hours))
        slot['load_hours'] += hours
    return dict(format='darsia-production-static-whole-run-assignment-v1', assignments=prepared,
                host_measurements=facts, ordering='descending_locked_frame_count_then_run_id',
                assignment_only=True, no_shared_nas_queue=True, production_permission_granted=False)


class HostRunStream:
    """Independent worker run states; ONE bounded host-wide next-run copier."""
    def __init__(self, fresh_root, *, host, assignment, io, stages=(), common_bindings=(), nas_is_network=None):
        if (assignment.get('format') != 'darsia-production-static-whole-run-assignment-v1'
                or assignment.get('assignment_only') is not True or assignment.get('production_permission_granted') is not False
                or host not in {'Main','Olav'}):
            raise ValueError('Predeclared non-GO whole-run assignment required')
        rows = [row for row in assignment['assignments'] if row['host'] == host]
        if not rows or len({row['run'] for row in assignment['assignments']}) != len(assignment['assignments']):
            raise ValueError('One host/worker per whole AC run required')
        self.host, self.assignment = host, deepcopy(assignment)
        self.root = Path(fresh_root).absolute()
        if str(self.root).startswith(('\\\\','//')) or self.root.resolve() != self.root:
            raise ValueError('Fresh local host ownership root required')
        self.root.mkdir(parents=False, exist_ok=False)
        self._copy_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='host-next-run-copy')
        self._allowed = {row['run']:row['worker'] for row in rows}
        self._ordered = {worker:[row['run'] for row in rows if row['worker']==worker]
                         for worker in set(self._allowed.values())}
        self._position = {worker:0 for worker in self._ordered}
        self.workers = {worker:RunStream(self.root/f'worker_{worker}',io=io,stages=stages,
            common_bindings=common_bindings,nas_is_network=nas_is_network,copy_executor=self._copy_executor)
            for worker in sorted(set(self._allowed.values()))}

    def _worker(self, worker, run):
        if type(worker) is not int or self._allowed.get(run) != worker:
            raise ValueError('Run differs from its predeclared whole-run host/worker assignment')
        return self.workers[worker]

    def start(self, worker, run, bindings, *, boundary_guard):
        owner = self._worker(worker,run)
        order, index = self._ordered[worker], self._position[worker]
        if index >= len(order) or run != order[index]:
            raise ValueError('Run differs from the predeclared next whole-run order')
        return owner.start(run,bindings,boundary_guard=boundary_guard)

    def copy_next(self, worker, run, bindings, *, before_copy):
        owner = self._worker(worker,run)
        order, index = self._ordered[worker], self._position[worker]
        if index >= len(order) or owner.active != order[index] or index+1 >= len(order) or run != order[index+1]:
            raise ValueError('Copy must be the one predeclared immediately next whole run')
        return owner.copy_next(run,bindings,before_copy=before_copy)

    def finish(self, worker, run, *, workers_retired_guard):
        owner = self._worker(worker,run)
        order, index = self._ordered[worker], self._position[worker]
        if index >= len(order) or run != order[index]:
            raise ValueError('Retirement differs from the current declared whole run')
        value = owner.finish(run,workers_retired_guard=workers_retired_guard)
        self._position[worker] += 1
        return value

    def close(self):
        self._copy_executor.shutdown(wait=True,cancel_futures=True)
