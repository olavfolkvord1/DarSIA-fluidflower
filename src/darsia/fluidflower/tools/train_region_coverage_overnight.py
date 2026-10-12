"""Detached, per-series regional coverage experiment; never replaces a model.

Missed visible aqueous CO2 is dense aqueous supervision, as clarified by the
reviewer on 2026-09-20. Original documents and mixed labels remain unchanged.
"""
from __future__ import annotations
from darsia.fluidflower.review_storage import resolve_review_path

from darsia.fluidflower.tools._paths import project_path, source_path, process_directory, series_root, series_path, tool_path, local_cache_root, package_root
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '1')
os.environ['PHASE_PROPOSAL_PROCESSES'] = '0'
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import cv2
import numpy as np
from threadpoolctl import threadpool_limits
from darsia.fluidflower import phase_optical_regions as regional, phase_runtime
from darsia.fluidflower.phase_training import logistic, joint_aqueous
from darsia.fluidflower.phase_workflow import discover_phase_review_sessions
from darsia.fluidflower.project import sha256_file
from darsia.fluidflower.tools import train_phase_optical_regions as extraction
from darsia.fluidflower.tools.ffac_v5_geometry_policy import (
    verify_accepted_geometry as verify_run_geometry,
)
from darsia.fluidflower.tools.train_phase_model_further import capture_inputs, read, native_source
from darsia.fluidflower.tools.render_concentration_movie import write

NAS = Path(r'\\UNAS-Pro-8\UiB_FluidFlower')
REPO = series_root("ffac")
COLUMNS = np.array([4,5,6,7,8,14,15,16,17,18])


def emit(root, stage, **extra):
    value = dict(stage=stage, updated_utc=datetime.now(timezone.utc).isoformat(), **extra)
    write(root/'overnight_status.json', value)
    print(json.dumps(value), flush=True)


def configuration(series, stamp):
    project = NAS / ('FF_'+series[2:].upper()) / 'DarSIA'
    parent = project/'phase_models'/f'phase-model-20260916T210559Z-{series}-regional'
    if series == 'ffum':
        parent = project/'models/phase-model-20260920-ffum-color-regions'
    if series == 'ffbm':
        # The historical parent is a C1 pilot, not the complete BM series.
        expected = {'c1', 'c2', 'c3', 'c4', 'c5'}
        complete = []
        for receipt in (project/'models').glob('*/scope_receipt.json'):
            ready = receipt.parent/'studio_preparation.json'
            if (set(read(receipt).get('expected_runs', [])) == expected and ready.is_file()
                    and read(ready).get('presentation_ready')):
                complete.append(receipt)
        if not complete:
            raise ValueError('FF_BM requires a completed C1-C5 review source; refusing the C1-only pilot')
        parent = max(complete, key=lambda p:p.stat().st_mtime).parent
    index = read(local_cache_root() / 'phase_review_collections/index.json')
    old_id = f'phase-model-20260916T210559Z-{series}-regional'
    collections = [Path(v['manifest']['path']).parent for v in index['collections'] if v['source_model_id']==old_id]
    collection = collections[0] if len(collections)==1 else parent/'review_source'
    if series == 'ffbm':
        collection = parent/'review_source'
        from darsia.fluidflower.tools.extend_optical_region_studio import require_run_set
        require_run_set(read(collection/'postflight_manifest.json')['jobs'], expected)
    return dict(series=series, parent=str(parent), collection=str(collection),
        root=str(project/'results'/f'region-coverage-{stamp}'),
        model_root=str(project/'models'/f'phase-model-{stamp}-{series}-full-aqueous'),
        cache=str(local_cache_root() / 'region-coverage'/stamp/series))


def capture(config):
    root, parent = Path(config['root']), Path(config['parent'])
    root.mkdir(parents=True, exist_ok=True)
    evidence = root/'evidence.json'
    if evidence.exists():
        return read(evidence)
    emit(root, 'capturing', series=config['series'])
    _, pairs, regions, audit = capture_inputs(parent,config['series'],root/'evidence_capture',
        discover_phase_review_sessions(),preserve_native_revisions=True)
    value = dict(parent=str(parent), pairs=[[str(p) for p in pair] for pair in pairs],
        regions=[asdict(r) for r in regions], audit=audit)
    write(evidence, value)
    corrected=[r.region_id for r in regions if r.source=='manual' and
               r.payload.get('category')=='missed_visible_aqueous_co2']
    write(root/'supervision_policy.json', dict(
        policy='missed-aqueous-is-dense-reviewer-clarification-20260920',
        reviewer_statement='The whole marked area is aqueous CO2',
        previous_interpretation='aqueous_present_region',current_interpretation='aqueous',
        corrected_regions=corrected,original_documents_modified=False,
        mixed_phase_classes_unchanged=True))
    write(root/'job.json', config)
    return value


def context_catalog(evidence, collection):
    """Gather only explicitly referenced context archives, including ancestors."""
    candidates = {Path(evidence['parent'])}
    candidates.update(Path(v['root']) for v in evidence['audit']['family_models'].values())
    for j in read(collection/'postflight_manifest.json')['jobs']:
        if j.get('raw_phase_map_output_dir'):
            candidates.add(Path(j['raw_phase_map_output_dir']).parent.parent)
    # Original review candidates can refer to contexts outside the direct lineage.
    for _, manifest in evidence['pairs']:
        for j in read(Path(manifest))['jobs']:
            if j.get('raw_phase_map_output_dir'):
                candidates.add(Path(j['raw_phase_map_output_dir']).parent.parent)
    result = []
    for candidate in sorted(candidates):
        if not (candidate/'summary.json').is_file():
            continue
        profile = phase_runtime.load_physics_profile(candidate)
        for key, entry in (profile or {}).get('runs',{}).items():
            path = candidate/entry['path']
            if path.is_file():
                result.append(dict(key=key, root=str(candidate), **entry, archive=str(path)))
    return result


def _verified_grid(rows, path):
    """Select one grid only when every inherited copy has identical bytes."""
    if not rows:
        raise ValueError(f'No exact inherited two-arm physical grid: {path}')
    for row in rows:
        if sha256_file(Path(row['archive'])) != row['sha256']:
            raise ValueError(f'Changed context: {path}')
    digests = {row['sha256'] for row in rows}
    if len(digests) != 1:
        raise ValueError(f'Ambiguous inherited two-arm physical grids: {path}')
    return min(rows, key=lambda row: str(row['archive']).casefold())


def _legacy_ffac_facies_binding(job, native, path, catalog):
    """Use the original non-resampled rig domain after physical-grid verification."""
    facies_path = NAS/'Albus/Results'/job['run']/'setup/rig/facies.npz'
    with np.load(facies_path, allow_pickle=True) as source:
        mask = source['array'] != 0
        spacing = np.asarray(source['metadata'].item()['dimensions']) / mask.shape
    if mask.shape != tuple(native['analysis_image_shape_hw']):
        raise ValueError(f'Legacy FF_AC rig and native canvas differ: {path}')
    key = f"{job['dataset']}/{job['run']}"
    choices = [row for row in catalog if row['key'] == key]
    if not choices:
        raise ValueError(f'No registered physical grid verifies legacy facies: {path}')
    representatives = {}
    for row in sorted(choices, key=lambda item: (item['sha256'], str(item['archive']).casefold())):
        representatives.setdefault(row['sha256'], row)
    matching = []
    for row in representatives.values():
        archive = Path(row['archive'])
        if sha256_file(archive) != row['sha256']:
            raise ValueError(f'Changed context: {path}')
        with np.load(archive) as registered:
            registered_spacing = (abs(float(registered['y_m'][1]-registered['y_m'][0])),
                                  abs(float(registered['x_m'][1]-registered['x_m'][0])))
            if (np.array_equal(registered['domain'], mask)
                    and np.allclose(registered_spacing, spacing, rtol=1e-5)):
                matching.append(row)
    if not matching:
        raise ValueError(f'Legacy facies differ from registered physical grid: {path}')
    quiz = native.get('phase_quiz_domain')
    return dict(kind='legacy-ffac-facies', path=str(facies_path),
        sha256=sha256_file(facies_path), spacing_m=spacing.tolist(),
        geometry='original full-resolution shape-corrected rig; no additional resampling',
        excluded_facies=[0], domain_fraction=float(mask.mean()),
        saved_quiz_sampling_domain=quiz,
        domain_scope='Porous inference domain; quiz proposal sampling may use additional valid-mask exclusions',
        verified_physical_grid=matching[0]['archive'],
        verified_physical_grid_sha256=matching[0]['sha256'],
        matching_physical_grid_sha256=sorted({row['sha256'] for row in matching}))


def _v5_nominal_spacing(archive):
    """Use only for regional pixel sampling; exact metric coordinates stay 2D."""
    mask = archive['domain'].astype(bool)
    x_native, y_native = archive['x_m_native'], archive['y_m_native']
    height, width = mask.shape
    rr = np.arange(0, height, max(1, height // 40), dtype=np.int64)
    cc = np.arange(0, width, max(1, width // 60), dtype=np.int64)
    rr = rr[rr + 1 < height]
    cc = cc[cc + 1 < width]
    grid = np.ix_(rr, cc)
    valid_x = mask[grid] & mask[np.ix_(rr, cc + 1)]
    valid_y = mask[grid] & mask[np.ix_(rr + 1, cc)]
    dx = np.hypot(
        x_native[np.ix_(rr, cc + 1)] - x_native[grid],
        y_native[np.ix_(rr, cc + 1)] - y_native[grid])
    dy = np.hypot(
        x_native[np.ix_(rr + 1, cc)] - x_native[grid],
        y_native[np.ix_(rr + 1, cc)] - y_native[grid])
    if (not np.any(valid_x) or not np.any(valid_y)
            or not np.isfinite(dx[valid_x]).all()
            or not np.isfinite(dy[valid_y]).all()
            or np.any(dx[valid_x] <= 0) or np.any(dy[valid_y] <= 0)):
        raise ValueError('V5 run-native geometry has no finite positive local pixel spacing')
    return mask, (float(np.median(dy[valid_y])), float(np.median(dx[valid_x])))


def _verify_two_arm_equivalence(equivalent, path, digest, native):
    source = Path(equivalent['source_manifest_path']).resolve()
    target = Path(equivalent['materialized_manifest_path']).resolve()
    if (target != Path(path).resolve()
            or sha256_file(target) != digest
            or sha256_file(source) != equivalent['source_manifest_sha256']):
        raise ValueError(f'Two-arm native source equivalence changed: {path}')
    source_native = read(source)
    for field in ('dataset', 'run', 'canvas_wh', 'content_box_xywh',
                  'analysis_image_shape_hw', 'additional_spatial_resampling'):
        if source_native.get(field) != native.get(field):
            raise ValueError(f'Two-arm physical geometry changed: {path}/{field}')


def bind_domains(evidence, collection):
    """Resolve each annotation's original geometry, never last-per-run geometry."""
    catalog = context_catalog(evidence, collection)
    audit = evidence.get('audit') or {}
    v5_binding = audit.get('v5_raw_source_binding')
    partition_binding = audit.get('v5_review_fit_partition')
    chemistry_scope = audit.get('v5_chemistry_fit_scope')
    fit_partition_rows = None
    excluded_runs = set()
    if partition_binding is not None:
        if v5_binding is None:
            raise ValueError('Track1 review partition requires V5 source binding')
        from darsia.fluidflower.phase_training.v5_review_fit_partition import (
            verify_partition_rows,
        )
        fit_partition_rows = verify_partition_rows(partition_binding,
                                                   evidence['regions'])
    if v5_binding is not None:
        if chemistry_scope is None:
            raise ValueError('V5 regional source lacks chemistry fit exclusions')
        from darsia.fluidflower.phase_training.v5_chemistry_fit_scope import verify_scope
        excluded_runs = verify_scope(chemistry_scope, evidence['regions'])
    elif chemistry_scope is not None:
        raise ValueError('Chemistry fit exclusions lack V5 source binding')
    plan = extraction.groups(evidence, excluded_runs=excluded_runs,
                             fit_partition_rows=fit_partition_rows)
    bindings = {}
    two_arm = (evidence.get('audit') or {}).get('two_arm_source_binding') or {}
    equivalents = {row['materialized_manifest_sha256']: row
                   for row in two_arm.get('source_equivalences') or []}
    if len(equivalents) != len(two_arm.get('source_equivalences') or []):
        raise ValueError('Ambiguous two-arm physical source equivalence')
    for g in plan:
        native, job, path = g['native'], g['job'], g['native_path']
        if str(path) in bindings:
            continue
        digest = sha256_file(path)
        key = f"{job['dataset']}/{job['run']}"
        exact = [r for r in catalog if r['key']==key and r['native_manifest_sha256']==digest]
        if native.get('v5_photometry', {}).get('format') == 'ffac-v5-native-source-materialization-v1':
            if job['dataset'] != 'ffac' or digest not in equivalents:
                raise ValueError(f'V5 native source lacks an exact original-source equivalence: {path}')
            equivalent = equivalents[digest]
            _verify_two_arm_equivalence(equivalent, path, digest, native)
            if native.get('v5_track1_photo_geometry') is not None:
                if native.get('v5_run_geometry') is not None or native.get('v5_frame_geometry') is not None:
                    raise ValueError('Track1 optical source inherited physical geometry')
                from darsia.fluidflower.tools.ffac_v5_track1_photo_context import (
                    Track1FramePhotoCohort,
                )
                photo = Track1FramePhotoCohort(path)
                if (photo.source_manifest_sha256 != equivalent['source_manifest_sha256']
                        or len(photo.rows) != native['frame_count']):
                    raise ValueError('Track1 photo source differs from exact review revision')
                photo.frame(0)
                bindings[str(path)] = dict(
                    kind='v5-track1-photo-only',
                    source_native_sha256=photo.source_manifest_sha256,
                    materialized_native_sha256=digest,
                    photo_geometry_contract=native['v5_track1_photo_geometry']['contract'],
                    frame_photo_receipts=len(photo.rows),
                    original_rig_facies_preserved=False,
                    physical_coordinates_2d=False,
                    nominal_spacing_only_for_regional_pixel_sampling=True,
                    model_scope='studio-validation-only-no-mass',
                )
            else:
                geometry = verify_run_geometry(
                    Path(equivalent['source_manifest_path']), Path(path))
                bindings[str(path)] = dict(
                    kind='v5-native-2d-context', path=geometry['context_npz_path'],
                    sha256=geometry['context_npz_sha256'],
                    source_native_sha256=equivalent['source_manifest_sha256'],
                    materialized_native_sha256=digest,
                    geometry_receipt_path=geometry['receipt_path'],
                    geometry_receipt_sha256=geometry['receipt_sha256'],
                    original_rig_facies_preserved=False,
                    physical_coordinates_2d=True,
                    nominal_spacing_only_for_regional_pixel_sampling=True,
                )
        elif job['dataset']=='ffac' and native.get('additional_spatial_resampling') is False:
            if digest in equivalents:
                _verify_two_arm_equivalence(equivalents[digest], path, digest, native)
            bindings[str(path)] = _legacy_ffac_facies_binding(job, native, path, catalog)
            if digest in equivalents:
                bindings[str(path)].update(source_native_sha256=equivalents[digest]['source_manifest_sha256'],
                    materialized_native_sha256=digest, original_rig_facies_preserved=True)
        elif exact:
            record = _verified_grid(exact, path)
            bindings[str(path)] = dict(kind='registered-context', path=record['archive'],sha256=record['sha256'])
        elif digest in equivalents:
            equivalent = equivalents[digest]
            _verify_two_arm_equivalence(equivalent, path, digest, native)
            inherited = [r for r in catalog if r['key'] == key and
                         r['native_manifest_sha256'] == equivalent['source_manifest_sha256']]
            record = _verified_grid(inherited, path)
            bindings[str(path)] = dict(kind='registered-context', path=record['archive'],
                sha256=record['sha256'], source_native_sha256=equivalent['source_manifest_sha256'],
                materialized_native_sha256=digest, physical_grid_bytes_preserved=True)
        elif native.get('domain',{}).get('path') or job.get('phase_transfer_context') or job['dataset']=='ffum':
            descriptor = native.get('domain') or job.get('phase_transfer_context')
            # Legacy benchmark domain only stores a mask. Its physical grid
            # must come from an explicitly recorded context on the same shape.
            if descriptor and 'sha256' not in descriptor:
                choices = [r for r in catalog if r['key']==key]
                if not choices:
                    raise ValueError(f'No physical grid for legacy domain: {path}')
                archive=Path(choices[0]['archive'])
                with np.load(archive) as a, np.load(resolve_review_path(descriptor['path'])) as b:
                    if a['domain'].shape != b['domain'].shape or not np.array_equal(a['domain'],b['domain']):
                        raise ValueError(f'Legacy domain differs from physical grid: {path}')
                bindings[str(path)]=dict(kind='registered-context',path=str(archive),sha256=sha256_file(archive),
                    legacy_domain_path=descriptor['path'],legacy_domain_sha256=sha256_file(resolve_review_path(descriptor['path'])))
            else:
                domain, spacing, receipt = extraction.native_domain(job,native)
                bindings[str(path)] = dict(kind='native-domain', **receipt, spacing_m=list(spacing))
                del domain
        else:
            raise ValueError(f'No exact registered training domain for {key}: {path}')
    if partition_binding is not None:
        # The fit plan deliberately omits evaluation-only reviews before any
        # image extraction. The candidate receipt still inventories all 52
        # original source revisions, including those with zero fit rows.
        # Bind their small manifests/SHA only; never open their RAW or JPEG.
        from darsia.fluidflower.tools.ffac_v5_track1_photo_context import (
            COHORT_CONTRACT as TRACK1_COHORT, Track1FramePhotoCohort,
        )
        for equivalent in equivalents.values():
            path = Path(equivalent['materialized_manifest_path'])
            if str(path) in bindings:
                continue
            digest = sha256_file(path)
            native = read(path)
            marker = native.get('v5_track1_photo_geometry') or {}
            if (digest != equivalent['materialized_manifest_sha256']
                    or marker.get('contract') != TRACK1_COHORT
                    or native.get('v5_run_geometry') is not None
                    or native.get('v5_frame_geometry') is not None):
                raise ValueError(f'Evaluation-only Track1 revision lost source binding: {path}')
            photo = Track1FramePhotoCohort(path)
            if (photo.source_manifest_sha256 != equivalent['source_manifest_sha256']
                    or len(photo.rows) != native['frame_count']):
                raise ValueError(f'Evaluation-only Track1 source revision changed: {path}')
            bindings[str(path)] = dict(
                kind='v5-track1-photo-only',
                source_native_sha256=photo.source_manifest_sha256,
                materialized_native_sha256=digest,
                photo_geometry_contract=TRACK1_COHORT,
                frame_photo_receipts=len(photo.rows),
                original_rig_facies_preserved=False,
                physical_coordinates_2d=False,
                nominal_spacing_only_for_regional_pixel_sampling=True,
                model_scope='studio-validation-only-no-mass',
                evaluation_only_no_extraction=True,
            )
    return plan, bindings


def extract(config, evidence, plan, bindings):
    root, cache = Path(config['root']), Path(config['cache'])
    cache.mkdir(parents=True, exist_ok=True)
    original = extraction.native_domain
    by_native = {json.dumps(g['native'],sort_keys=True):bindings[str(g['native_path'])] for g in plan}
    # The sorted plan revisits one rig facies file for many reviewed frames.
    # Decode its exact SHA-bound bytes once and share a read-only domain until
    # the source changes. Check file metadata on every frame and rehash before
    # moving to another source and at successful extraction completion. This
    # avoids hundreds of duplicate NAS reads without accepting a changed rig.
    domain_snapshot = {}
    def verified_source(path, expected):
        if sha256_file(path) != expected:
            raise ValueError('Training domain changed after preflight')
    def signature(path):
        stat = path.stat()
        return stat.st_size, stat.st_mtime_ns
    def domain(job, native):
        binding = by_native[json.dumps(native,sort_keys=True)]
        if binding['kind']=='v5-track1-photo-only':
            raise ValueError('Track1 regional extraction must use per-frame photo support')
        if binding['kind']=='native-domain':
            return original(job,native)
        path = Path(binding['path'])
        key = (str(path), binding['sha256'], binding['kind'])
        current_signature = signature(path)
        if domain_snapshot.get('key') == key and domain_snapshot['signature'] == current_signature:
            return domain_snapshot['mask'], domain_snapshot['spacing'], binding
        if domain_snapshot:
            verified_source(domain_snapshot['path'], domain_snapshot['sha256'])
        # Hash the bytes we actually decode. A later metadata check and the
        # end-of-source SHA guard close the source-mutation window.
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != binding['sha256'] or signature(path) != current_signature:
            raise ValueError('Training domain changed after preflight')
        with np.load(io.BytesIO(raw), allow_pickle=binding['kind']=='legacy-ffac-facies') as a:
            if binding['kind']=='legacy-ffac-facies':
                mask = a['array'] != 0
                spacing = tuple(binding['spacing_m'])
            elif binding['kind']=='v5-native-2d-context':
                mask, spacing = _v5_nominal_spacing(a)
            else:
                mask = a['domain'].astype(bool)
                spacing = (abs(float(a['y_m'][1]-a['y_m'][0])),
                           abs(float(a['x_m'][1]-a['x_m'][0])))
        if mask.shape!=tuple(native['analysis_image_shape_hw']):
            raise ValueError('Training domain and source image differ')
        mask.setflags(write=False)
        domain_snapshot.clear()
        domain_snapshot.update(key=key, path=path, sha256=binding['sha256'],
                               signature=current_signature, mask=mask, spacing=spacing)
        return mask,spacing,binding
    extraction.native_domain = domain
    try:
        emit(root,'extracting',frames=len(plan),regions=len(evidence['regions']))
        result = extraction.extract(root,cache,threads=config.get('cpu_threads'),
                                    feature_cache_root=config.get('feature_cache_root'),backend=config.get('backend','cuda'))
        if domain_snapshot:
            verified_source(domain_snapshot['path'], domain_snapshot['sha256'])
        return result
    finally:
        extraction.native_domain = original


def _frozen_v5_fit_policies(evidence):
    """Recheck the source-bound policies before sealing fitted head metadata."""
    audit=evidence.get('audit') or {}
    v5_binding=audit.get('v5_raw_source_binding')
    chemistry_scope=audit.get('v5_chemistry_fit_scope')
    partition_binding=audit.get('v5_review_fit_partition')
    if v5_binding is not None:
        if chemistry_scope is None:
            raise ValueError('V5 regional fit omitted frozen chemistry exclusion')
        from darsia.fluidflower.phase_training.v5_chemistry_fit_scope import verify_scope
        verify_scope(chemistry_scope,evidence['regions'])
        if partition_binding is not None:
            from darsia.fluidflower.phase_training.v5_review_fit_partition import verify_partition_rows
            verify_partition_rows(partition_binding,evidence['regions'])
    elif chemistry_scope is not None or partition_binding is not None:
        raise ValueError('V5 fit policy without source-bound float input')
    return chemistry_scope,partition_binding


def fit(config, evidence, extracted):
    root=Path(config['root'])
    if sha256_file(Path(extracted['samples']))!=extracted['samples_sha256']:
        raise ValueError('Training samples changed')
    with np.load(extracted['samples']) as a:
        matrix=a['features'][:,COLUMNS];owners=a['owners']
    regions=[logistic.ReviewedRegion(**r) for r in evidence['regions']]
    chemistry_scope,partition_binding=_frozen_v5_fit_policies(evidence)
    seeds={};seed_reports={}
    for head in joint_aqueous.HEADS:
        emit(root,'fitting_seed',head=head,rows=len(matrix))
        seeds[head],seed_reports[head]=logistic.fit_weighted_logistic_mil(matrix,owners,regions,
            head=head,runs=None,regularization=.001,backend=config.get('backend','cuda'),max_iter=1500)
    results={}
    for name in ('full_aqueous',):
        output=root/name
        output.mkdir(exist_ok=True)
        emit(root,'fitting_dense_aqueous',all_sampled_dense_pixels_supervised=True)
        fits,report=joint_aqueous.fit_joint_aqueous_mil(matrix,owners,regions,seed_fits=seeds,
            regularization={head:.001 for head in seeds},backend=config.get('backend','cuda'),max_iter=2000,tolerance=1e-6)
        packed={}
        for head,fitted in fits.items():
            values=asdict(fitted)
            for field in ('mean','scale','coefficients'):
                full=np.ones(len(regional.FEATURE_NAMES)) if field=='scale' else np.zeros(len(regional.FEATURE_NAMES))
                full[COLUMNS]=values[field];values[field]=full.tolist()
            packed[head]=values
        protocol=dict(series=config['series'],dense_aqueous_corrections=True,
            small_region_sampling='native-points-with-interpolated-regional-features',
            original_documents_modified=False,mixed_phase_classes_unchanged=True,
            all_sampled_dense_pixels_supervised=True,other_regional_positive_fraction=.1,
            context_maps_used=False,independent_validation=False,backend=config.get('backend','cuda'),cpu_threads=config.get('cpu_threads',8),
            evidence_sha256=sha256_file(root/'evidence.json'),extraction_sha256=sha256_file(root/'extraction.json'))
        write(output/'protocol.json',protocol)
        model=dict(format=regional.VERSION,feature_names=list(regional.FEATURE_NAMES),
            settings=read(root/'protocol.json')['settings'],heads=packed,parent_model=evidence['parent'],
            evidence_path=str(root/'evidence.json'),evidence_sha256=sha256_file(root/'evidence.json'),
            protocol_sha256=sha256_file(output/'protocol.json'),fit_report=report,seed_reports=seed_reports,
            scientific_acceptance=False,training_scope=f"effective {config['series']} native reviews",
            experiment='dense-aqueous-corrections',dense_aqueous_corrections=True)
        if chemistry_scope is not None:
            model['v5_chemistry_fit_scope']=dict(chemistry_scope)
        if partition_binding is not None:
            model['v5_review_fit_partition']=dict(partition_binding)
        write(output/'regional_model.json',model)
        results[name]=dict(path=str(output),converged=report['converged'],objective=report['objective'])
    write(root/'fit_comparison.json',results)
    if not results['full_aqueous']['converged']:
        raise ValueError('Coverage candidate did not converge; checkpoint retained for inspection')
    return root/'full_aqueous'


def run_series(config, capture_only=False, studio_only=False, *, prepare_disagreements=False):
    root=Path(config['root']);root.mkdir(parents=True,exist_ok=True)
    try:
        if not studio_only:
            evidence=capture(config)
            plan,bindings=bind_domains(evidence,Path(config['collection']))
            write(root/'domain_bindings.json',bindings)
            emit(root,'preflight_ready',regions=len(evidence['regions']),frames=len(plan),native_revisions=len(bindings))
            if capture_only:return
            cv2.setNumThreads(8)
            with threadpool_limits(limits=8):
                extracted=extract(config,evidence,plan,bindings)
                fitted=fit(config,evidence,extracted)
            emit(root,'trained',experiment=str(fitted))
            return
        fitted=root/'full_aqueous'
        if not (fitted/'regional_model.json').is_file() or not read(fitted/'regional_model.json')['fit_report']['converged']:
            return
        # One GPU-owning process at a time; no competing per-series training.
        commands = [
            ('darsia.fluidflower.tools.prepare_optical_region_studio',['--experiment',str(fitted),'--collection',config['collection'],'--root',config['model_root'],'--fresh-optical','--workers','1']),
            ('darsia.fluidflower.tools.finish_optical_region_studio',['--root',config['model_root']])]
        if prepare_disagreements:
            commands.append(('darsia.fluidflower.tools.build_phase_disagreement_review',
                ['--model',config['model_root'],'--output',str(root/'disagreements.json')]))
        for module,args in commands:
            emit(root,'preparing_studio',step=module)
            subprocess.run([sys.executable,'-m',module,*args],cwd=process_directory(),check=True,
                           stdout=sys.stdout,stderr=sys.stderr)
        if not prepare_disagreements:
            write(root/'disagreements_deferred.json',dict(status='deferred',
                  action='Check disagreements',model_root=config['model_root']))
        emit(root,'ready_for_review',model_root=config['model_root'],
             disagreements_deferred=not prepare_disagreements)
    except Exception as exc:
        emit(root,'failed',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        gc.collect()
        try:
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
        except ImportError:pass


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stamp',required=True)
    parser.add_argument('--series',nargs='+',default=['ffum','ffac','ffbm'],choices=['ffum','ffac','ffbm'])
    parser.add_argument('--capture-only',action='store_true')
    parser.add_argument('--prepare-disagreements',action='store_true',
                        help='Build disagreement list now instead of on request in Studio.')
    args=parser.parse_args()
    # Fit every series before spending time on full-resolution review movies.
    for studio_only in ([False] if args.capture_only else [False,True]):
        for series in args.series:
            config=configuration(series,args.stamp)
            root=Path(config['root']);root.mkdir(parents=True,exist_ok=True)
            snapshot=root/'code_snapshot';snapshot.mkdir(exist_ok=True)
            for module in (Path(__file__),Path(extraction.__file__),Path(logistic.__file__),
                           Path(joint_aqueous.__file__),Path(regional.__file__),
                           project_path('scripts/prepare_optical_region_studio.py'),project_path('scripts/finish_optical_region_studio.py')):
                shutil.copy2(module,snapshot/module.name)
            with (root/('preflight.log' if args.capture_only else 'overnight.log')).open('a',encoding='utf-8',buffering=1) as log:
                old=sys.stdout;olderr=sys.stderr
                sys.stdout=sys.stderr=log
                try:run_series(config,args.capture_only,studio_only,
                               prepare_disagreements=args.prepare_disagreements)
                finally:sys.stdout=old;sys.stderr=olderr
            print(series,read(root/'overnight_status.json')['stage'],flush=True)


if __name__=='__main__':main()
