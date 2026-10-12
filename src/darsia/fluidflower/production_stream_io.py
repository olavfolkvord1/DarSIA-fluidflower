"""Opt-in staged-input identity reads; no admission, science, CLI or deletion.

The coordinator hashes each staged file once and pins its local file identity.
Readers subsequently check that identity and the original full logical binding.
A changed file is an error, never an invitation to rehash/requalify it. Source
qualification, aliases, native input guards and frame-boundary authority remain
in the original ManifestAssetReader and native caller.
"""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager, ExitStack
from copy import deepcopy
import json
import ntpath
import os
from pathlib import Path
import re
import stat
import threading


class StageIdentityError(ValueError):
    pass


def _key(path):
    value = os.fspath(path)
    if os.name == 'nt' or ntpath.splitdrive(value)[0] or value.startswith(('\\\\', '//')):
        return ntpath.normcase(ntpath.normpath(ntpath.abspath(value)))
    return os.path.normpath(os.path.abspath(value))


def _full(value):
    if (not isinstance(value, Mapping) or set(value) != {'path', 'sha256', 'size_bytes'}
            or not isinstance(value['path'], str) or not (os.path.isabs(value['path']) or ntpath.isabs(value['path']))
            or not isinstance(value['sha256'], str) or re.fullmatch('[0-9a-fA-F]{64}', value['sha256']) is None
            or type(value['size_bytes']) is not int or value['size_bytes'] < 0):
        raise StageIdentityError('Full absolute SHA/integer-size descriptor required')
    return dict(value, sha256=value['sha256'].lower())


def _same(a, b):
    a, b = _full(a), _full(b)
    return _key(a['path']) == _key(b['path']) and (a['sha256'], a['size_bytes']) == (b['sha256'], b['size_bytes'])


def _stamp_stat(value):
    if not stat.S_ISREG(value.st_mode):
        raise StageIdentityError('Ordinary staged file required')
    return dict(size_bytes=value.st_size, mtime_ns=value.st_mtime_ns,
                ctime_ns=value.st_ctime_ns, device=value.st_dev, file_id=value.st_ino)


def local_identity(path):
    path = Path(path)
    if path.is_symlink() or path.resolve() != path.absolute():
        raise StageIdentityError('Staged locator must not contain a symlink or reparse redirection')
    return _stamp_stat(path.stat())


def _open_matches(value, pinned):
    # CPython 3.13 Windows Path.stat and fstat report different ctime semantics
    # for some freshly written files. PATH ctime is still checked strictly
    # before/after every open. The actual open handle must independently match
    # its stable NTFS file-ID/device/size/mtime, so replacement is not accepted.
    actual = _stamp_stat(value)
    fields = ('size_bytes', 'mtime_ns', 'device', 'file_id') if os.name == 'nt' else tuple(actual)
    return all(actual[key] == pinned[key] for key in fields)


def _local(path, cache_root):
    path, root = Path(path), Path(cache_root).resolve()
    if (str(root).startswith(('\\\\', '//')) or str(path).startswith(('\\\\', '//'))
            or not path.is_absolute() or not path.resolve().is_relative_to(root)
            or path.resolve() == root or path.resolve() != path.absolute()):
        raise StageIdentityError('Staged file must stay inside the exact local cache')
    return path


def verified_stage_receipt(manifest, staged_inputs, *, manifest_descriptor, cache_root, io):
    """ONE coordinator verification of already copied inputs; no array reads.

    Existing NasIO does SHA/stat verification, including its transient-only NAS
    policy. This function neither copies inputs nor opens their original NAS
    locators. The receipt should be written once by the existing coordinator
    with NasIO.write_json; workers receive its full SHA-bound descriptor.
    """
    bound = _full(manifest_descriptor)
    if io.read_json(bound, max_bytes=bound['size_bytes']) != manifest:
        raise StageIdentityError('Actual immutable manifest bytes differ')
    inputs = manifest.get('inputs')
    if (manifest.get('format') != 'darsia-production-job-v1' or not isinstance(inputs, Mapping)
            or not inputs or set(inputs) != set(staged_inputs)):
        raise StageIdentityError('Exactly the original flat input table must be staged')
    rows, verified = {}, {}
    for role, logical in inputs.items():
        logical, local = _full(logical), _full(staged_inputs[role])
        if (logical['sha256'], logical['size_bytes']) != (local['sha256'], local['size_bytes']):
            raise StageIdentityError('Local bytes differ from the original manifest binding')
        path = _local(local['path'], cache_root)
        before = local_identity(path)
        content_key = (_key(local['path']), local['sha256'], local['size_bytes'])
        if content_key in verified:
            if before != verified[content_key]:
                raise StageIdentityError('Aliased staged file changed after coordinator verification')
        else:
            io.verify(local)  # Exactly one SHA pass per unique local content file.
        after = local_identity(path)
        if before != after or after['size_bytes'] != logical['size_bytes']:
            raise StageIdentityError('Staged file changed during coordinator SHA verification')
        rows[role] = dict(logical=logical, local=local, local_identity=after)
        verified[content_key] = after
    return dict(format='darsia-production-verified-stage-v1', complete=True,
                manifest=bound, cache_root=str(Path(cache_root).resolve()), inputs=rows,
                payload_sha_verified_by_coordinator=True, arrays_decoded=False,
                subsequent_read_policy='pinned_local_identity_no_rehash',
                production_permission_granted=False)


def rebind_verified_stage(parent_receipt, manifest, staged_inputs, *,
                          manifest_descriptor, cache_root, io):
    """Metadata-only child receipt for the SAME input bytes and live identities.

    A per-thread manifest may change settings/job fields, but cannot change any
    logical/local input binding or reuse a changed local file. The original
    reader still checks the actual new code/environment/purpose/source review.
    This function grants no admission or qualification and does no payload SHA.
    """
    parent = _full(parent_receipt)
    old = io.read_json(parent, max_bytes=parent['size_bytes'])
    original_bound = _full(old['manifest'])
    original_manifest = io.read_json(original_bound, max_bytes=original_bound['size_bytes'])
    stage = VerifiedStage(parent, original_manifest, staged_inputs, cache_root=cache_root, io=io)
    new_bound = _full(manifest_descriptor)
    if (io.read_json(new_bound, max_bytes=new_bound['size_bytes']) != manifest
            or manifest.get('format') != 'darsia-production-job-v1'
            or set(manifest['inputs']) != set(original_manifest['inputs'])
            or any(not _same(manifest['inputs'][role], original_manifest['inputs'][role])
                   for role in manifest['inputs'])):
        raise StageIdentityError('Rebinding may not change any actual flat input binding')
    for item in stage.rows.values():
        stage.check(item['logical'])
    result = deepcopy(old)
    result['manifest'] = new_bound
    result['parent_verified_stage'] = parent
    result['rebind_metadata_only'] = True
    result['payload_sha_rechecked_during_rebind'] = False
    return result


class VerifiedStage:
    """Read a bound coordinator receipt; validate metadata/stat, not payload SHA."""
    def __init__(self, receipt, manifest, staged_inputs, *, cache_root, io):
        receipt = _full(receipt)
        value = io.read_json(receipt, max_bytes=receipt['size_bytes'])
        if (value.get('format') != 'darsia-production-verified-stage-v1'
                or value.get('complete') is not True
                or value.get('payload_sha_verified_by_coordinator') is not True
                or value.get('arrays_decoded') is not False
                or value.get('production_permission_granted') is not False
                or value.get('subsequent_read_policy') != 'pinned_local_identity_no_rehash'
                or _key(value['cache_root']) != _key(cache_root)
                or set(value['inputs']) != set(manifest['inputs'])
                or set(staged_inputs) != set(manifest['inputs'])):
            raise StageIdentityError('Exact coordinator stage receipt required')
        original = _full(value['manifest'])
        if io.read_json(original, max_bytes=original['size_bytes']) != manifest:
            raise StageIdentityError('Coordinator receipt names a different manifest')
        self.rows, self.references = {}, {}
        self._lock = threading.RLock()
        for role, row in value['inputs'].items():
            if set(row) != {'logical', 'local', 'local_identity'}:
                raise StageIdentityError('Unexpected staging row fields')
            if not _same(row['logical'], manifest['inputs'][role]) or not _same(row['local'], staged_inputs[role]):
                raise StageIdentityError('Stage receipt locator/content differs')
            path = _local(row['local']['path'], cache_root)
            if (local_identity(path) != row['local_identity']
                    or row['local_identity']['size_bytes'] != row['logical']['size_bytes']):
                raise StageIdentityError('Staged identity changed; no silent rehash is permitted')
            key = _key(row['logical']['path'])
            if key in self.rows and self.rows[key] != row:
                raise StageIdentityError('Conflicting staging receipt aliases')
            self.rows[key] = deepcopy(row)
        self.receipt_descriptor = deepcopy(receipt)

    def check(self, logical):
        key = _key(logical['path'])
        row = self.rows.get(key)
        if row is None or not _same(logical, row['logical']):
            raise StageIdentityError('Read binding is absent from the verified stage')
        path = Path(row['local']['path'])
        if local_identity(path) != row['local_identity']:
            raise StageIdentityError('Staged identity changed; no silent rehash is permitted')
        return path

    @contextmanager
    def hold(self, logical):
        """Track a worker/prefetch reference through its actual use interval."""
        path = self.check(logical)
        key = _key(path)
        with self._lock:
            self.references[key] = self.references.get(key, 0)+1
        try:
            yield path
            self.check(logical)
        finally:
            with self._lock:
                self.references[key] -= 1
                if not self.references[key]:
                    del self.references[key]

    def read_bytes(self, logical):
        with self.hold(logical) as path:
            row = self.rows[_key(logical['path'])]
            with path.open('rb') as stream:
                if not _open_matches(os.fstat(stream.fileno()), row['local_identity']):
                    raise StageIdentityError('Opened file identity differs from staged receipt')
                data = stream.read(logical['size_bytes']+1)
                if not _open_matches(os.fstat(stream.fileno()), row['local_identity']):
                    raise StageIdentityError('Open file changed during read')
            if len(data) != logical['size_bytes']:
                raise StageIdentityError('Staged read truncated or grew')
            return data


def npy_contracts(stage):
    """Read only verified NPY headers; never materialize pixel/model arrays.

    Native constructors still validate the scientifically expected grid/dtype.
    These contracts freeze the exact staged header for subsequent open checks.
    """
    import numpy as np
    rows = []
    for key, item in stage.rows.items():
        bound = item['logical']
        if Path(bound['path']).suffix.lower() != '.npy':
            continue
        with stage.hold(bound) as path:
            with path.open('rb') as stream:
                if not _open_matches(os.fstat(stream.fileno()), item['local_identity']):
                    raise StageIdentityError('NPY header file identity differs')
                version = np.lib.format.read_magic(stream)
                if version not in {(1, 0), (2, 0)}:
                    raise StageIdentityError('Only fixed numeric NPY v1/v2 headers are supported')
                parser = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
                shape, _, dtype = parser(stream, max_header_size=16384)
                if not _open_matches(os.fstat(stream.fileno()), item['local_identity']):
                    raise StageIdentityError('NPY header file changed')
            if dtype.hasobject:
                raise StageIdentityError('Object NPY is not a scientific read-only input')
            rows.append(dict(descriptor=deepcopy(bound), shape=list(shape), dtype=dtype.str))
    return rows


def stream_reader_class(original_reader):
    """Opt-in subclass; omit verified_staging_receipt to preserve old behavior.

    The original constructor still validates code, environment/restart, aliases,
    opaque data and exact source qualification. No scientific guard is replaced.
    Full NPY shape/dtype contracts are required for the opt-in array path.
    """
    class StreamManifestAssetReader(original_reader):
        def __init__(self, manifest, staged_inputs, *, verified_staging_receipt=None,
                     array_contracts=(), **kwargs):
            self._verified_stage = None
            self._array_contracts = {}
            self._stream_json = {}
            self._frame_stack = None
            self._frame_paths = set()
            if verified_staging_receipt is not None:
                if kwargs.get('io') is None:
                    raise StageIdentityError('Opt-in reads require the explicit original NasIO engine')
                self._verified_stage = VerifiedStage(verified_staging_receipt, manifest, staged_inputs,
                    cache_root=kwargs['cache_root'], io=kwargs['io'])
                for row in array_contracts:
                    if set(row) != {'descriptor', 'shape', 'dtype'}:
                        raise StageIdentityError('Exact NPY descriptor/shape/dtype contract required')
                    descriptor = _full(row['descriptor'])
                    self._verified_stage.check(descriptor)
                    shape = row['shape']
                    if not isinstance(shape, (list, tuple)) or not shape or any(type(v) is not int or v < 0 for v in shape):
                        raise StageIdentityError('Exact integer NPY shape required')
                    key = _key(descriptor['path'])
                    if key in self._array_contracts:
                        raise StageIdentityError('Duplicate NPY contract')
                    self._array_contracts[key] = dict(descriptor=descriptor, shape=tuple(shape), dtype=row['dtype'])
            super().__init__(manifest, staged_inputs, **kwargs)

        def _stage_logical(self, descriptor):
            logical = self.logical_descriptor(descriptor)
            canonical = self._canonical_key(logical['path'])
            return self._verified_stage.rows[canonical]['logical']

        def path(self, descriptor):
            if self._verified_stage is None:
                return super().path(descriptor)
            bound = self._stage_logical(descriptor)
            path = self._verified_stage.check(bound)
            if self._frame_stack is not None and _key(bound['path']) not in self._frame_paths:
                self._frame_stack.enter_context(self._verified_stage.hold(bound))
                self._frame_paths.add(_key(bound['path']))
            return path

        @contextmanager
        def frame_inputs(self, *, boundary_guard):
            """Keep worker references until calculation ends; fresh native guard.

            The guard is supplied by the original caller. It must check actual
            source/PASS/freeze/control at the frame boundary; this method does
            not grant or manufacture authority. Array objects must not outlive
            this declared worker use interval.
            """
            if not callable(boundary_guard) or self._frame_stack is not None:
                raise StageIdentityError('One actual guarded frame-input interval required')
            boundary_guard()
            try:
                with ExitStack() as stack:
                    self._frame_stack = stack
                    yield self
                    boundary_guard()
            finally:
                self._frame_stack = None
                self._frame_paths.clear()

        def sha256(self, logical_path):
            if self._verified_stage is None:
                return super().sha256(logical_path)
            logical = self._bindings[self._canonical_key(logical_path)]
            self.path(logical)
            return logical['sha256']

        def json(self, descriptor):
            if self._verified_stage is None:
                return super().json(descriptor)
            logical = self.logical_descriptor(descriptor)
            if (logical['sha256'], logical['size_bytes']) in self._opaque_contents:
                raise ValueError('Opaque evidence may be hashed but never parsed')
            canonical = self._stage_logical(descriptor)
            self.path(canonical)  # Check current identity even when parsed JSON is cached.
            key = (_key(canonical['path']), canonical['sha256'])
            if key not in self._stream_json:
                # The unchanged NasIO parser rejects duplicate/nonfinite JSON.
                # Passing the raw LOCAL Path does not request a second SHA.
                with self._verified_stage.hold(canonical) as path:
                    value = self.io.read_json(path, max_bytes=canonical['size_bytes'])
                if not isinstance(value, dict):
                    raise ValueError('Expected source-bound JSON object')
                self._stream_json[key] = value
            return self._stream_json[key]

        def array(self, descriptor):
            if self._verified_stage is None:
                return super().array(descriptor)
            logical = self.logical_descriptor(descriptor)
            if ((logical['sha256'], logical['size_bytes']) in self._opaque_contents
                    or self._canonical_key(logical['path']) in getattr(self, '_assurance_locators', set())):
                raise ValueError('Opaque/assurance evidence may not enter scientific arrays')
            bound = self._stage_logical(descriptor)
            contract = self._array_contracts.get(_key(bound['path']))
            if contract is None:
                raise StageIdentityError('Opt-in NPY read lacks its exact native shape/dtype contract')
            if self._frame_stack is None:
                raise StageIdentityError('NPY reads require the declared guarded worker-use interval')
            import numpy as np
            if np.dtype(contract['dtype']).hasobject or Path(bound['path']).suffix.lower() != '.npy':
                raise StageIdentityError('Read-only non-object NPY input required')
            path = self.path(bound)
            with self._verified_stage.hold(bound):
                value = np.load(path, mmap_mode='r', allow_pickle=False)
                if tuple(value.shape) != contract['shape'] or value.dtype != np.dtype(contract['dtype']):
                    raise StageIdentityError('NPY differs from exact native shape/dtype')
                if value.flags.writeable:
                    raise StageIdentityError('Staged scientific arrays must be read-only')
                return value

    return StreamManifestAssetReader
