"""Create a detached, provenance-bound annotation copy for a translation revision.

No Studio registration or training is performed. Unsupported or partly mapped
documents are rejected instead of clipping regions or silently retaining old
coordinates. Run ``python -m scripts.phase_rig_annotation_migration --help``.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping


SPACE = 'normalized_native_review_canvas_16x9'
FORMAT = 'phase-annotation-translation-copy-v1'
GEOMETRY_KEYS = ('coordinate_space', 'shape', 'points', 'x', 'y', 'width', 'height',
                 'frame_index', 'elapsed_h')


def _sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _read(path):
    raw = Path(path).read_bytes()
    return json.loads(raw.decode('utf-8-sig')), _sha_bytes(raw)


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{label} must be a finite number')
    return float(value)


def _integer(value, label):
    number = _number(value, label)
    if number != int(number) or number < 0:
        raise ValueError(f'{label} must be a nonnegative integer')
    return int(number)


def _hash(value, label):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError(f'{label} must be a lowercase SHA-256')
    return value


def _layout(manifest):
    canvas = manifest.get('canvas_wh')
    content = manifest.get('content_box_xywh')
    if not isinstance(canvas, (list, tuple)) or len(canvas) != 2 or not isinstance(content, (list, tuple)) or len(content) != 4:
        raise ValueError('Manifest needs canvas_wh and content_box_xywh')
    cw, ch = [_integer(v, 'canvas') for v in canvas]
    ox, oy, w, h = [_integer(v, 'content box') for v in content]
    if min(cw, ch, w, h) <= 0 or ox+w > cw or oy+h > ch:
        raise ValueError('Invalid canvas/content bounds')
    return cw, ch, ox, oy, w, h


def _frames(manifest):
    records = manifest.get('frame_sources')
    if not isinstance(records, list):
        raise ValueError('Manifest needs explicit frame_sources')
    result = {}
    for row in records:
        index = _integer(row.get('frame_index'), 'manifest frame_index')
        if index in result:
            raise ValueError(f'Ambiguous duplicate manifest frame {index}')
        if not row.get('source_name') or not row.get('source_path'):
            raise ValueError(f'Frame {index} needs source_name and source_path')
        _number(row.get('elapsed_h'), 'manifest elapsed_h')
        result[index] = row
    return result


def _same_path(a, b):
    # Metadata may cross Windows/UNC slash spelling; neither a basename nor time
    # alone is an adequate identity for a frame.
    return str(a).replace('\\', '/').casefold() == str(b).replace('\\', '/').casefold()


def _bind_receipts(receipt_document, source_manifests, target_manifests,
                   source_hashes, target_hashes):
    revision = receipt_document.get('geometry_revision_id')
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError('A named geometry_revision_id is required')
    if receipt_document.get('schema') != 1 or not isinstance(receipt_document.get('frames'), list):
        raise ValueError('Receipt document needs schema 1 and frames list')
    bound = {}
    target_indices = set()
    frame_tables = {}
    for run in source_manifests:
        if run not in target_manifests:
            raise ValueError(f'Missing target manifest for {run}')
        old, new = source_manifests[run], target_manifests[run]
        if str(old.get('run', '')).lower() != run or str(new.get('run', '')).lower() != run:
            raise ValueError(f'Manifest run mismatch for {run}')
        old_layout, new_layout = _layout(old), _layout(new)
        if old_layout[-2:] != new_layout[-2:]:
            raise ValueError('Translation migration cannot resize native content')
        frame_tables[run] = (_frames(old), _frames(new), old_layout, new_layout)
    for binding in receipt_document['frames']:
        run = str(binding.get('run', '')).lower()
        if run not in frame_tables:
            raise ValueError(f'Receipt has no matching manifest pair: {run}')
        if binding.get('source_manifest_sha256') != source_hashes[run] or binding.get('target_manifest_sha256') != target_hashes[run]:
            raise ValueError(f'Manifest hash mismatch for {run}')
        old_index = _integer(binding.get('source_frame_index'), 'source_frame_index')
        new_index = _integer(binding.get('target_frame_index'), 'target_frame_index')
        key = (run, old_index)
        if key in bound or (run, new_index) in target_indices:
            raise ValueError(f'Ambiguous frame mapping: {run} / {old_index}')
        old_frames, new_frames, old_layout, new_layout = frame_tables[run]
        if old_index not in old_frames or new_index not in new_frames:
            raise ValueError(f'Receipt frame absent from manifest: {key}')
        old_frame, new_frame = old_frames[old_index], new_frames[new_index]
        for frame in (old_frame, new_frame):
            if frame['source_name'] != binding.get('source_name') or not _same_path(frame['source_path'], binding.get('source_path', '')):
                raise ValueError(f'Frame source identity mismatch: {key}')
        receipt = binding.get('receipt') or {}
        if receipt.get('accepted') is not True or receipt.get('status') not in ('accepted', 'baseline_identity'):
            raise ValueError(f'Unaccepted geometry receipt: {key}')
        shift = receipt.get('shift_xy_native_px')
        if not isinstance(shift, (list, tuple)) or len(shift) != 2:
            raise ValueError('Receipt needs shift_xy_native_px=[dx,dy]')
        shift = [_number(v, 'native sampling displacement') for v in shift]
        shape = receipt.get('shape_hw')
        if not isinstance(shape, (list, tuple)) or list(shape) != [old_layout[5], old_layout[4]]:
            raise ValueError('Receipt shape_hw does not match native content')
        _hash(receipt.get('reference_hash'), 'reference_hash')
        _hash(receipt.get('policy_hash'), 'policy_hash')
        if receipt['status'] == 'baseline_identity':
            old_ref = source_manifests[run].get('reference') or {}
            new_ref = target_manifests[run].get('reference') or {}
            if shift != [0., 0.] or old_ref.get('timeline_frame_index') != old_index or new_ref.get('timeline_frame_index') != new_index:
                raise ValueError('baseline_identity requires zero shift and explicit reference frames')
        bound[key] = {'binding': deepcopy(binding), 'old_frame': old_frame, 'new_frame': new_frame,
                      'old_layout': old_layout, 'new_layout': new_layout, 'shift': shift}
        target_indices.add((run, new_index))
    return bound


def _points(annotation):
    if annotation.get('coordinate_space') != SPACE:
        raise ValueError('Unknown or missing annotation coordinate space')
    shape = annotation.get('shape')
    points = annotation.get('points') or []
    if points:
        if shape not in ('polygon', 'freehand') or len(points) < 3:
            raise ValueError('Polygon needs a supported shape and at least three points')
        if any(not isinstance(p, (list, tuple)) or len(p) != 2 for p in points):
            raise ValueError('Polygon points must be [x,y] pairs')
        result = [[_number(p[0], 'point x'), _number(p[1], 'point y')] for p in points]
        area = abs(sum(result[i][0]*result[(i+1)%len(result)][1] - result[(i+1)%len(result)][0]*result[i][1] for i in range(len(result))))/2
        if area <= 0:
            raise ValueError('Polygon has zero signed area')
        return result, True
    if shape not in ('rectangle', 'box'):
        raise ValueError('Only explicit rectangles or polygons are supported')
    x, y, w, h = [_number(annotation.get(k), k) for k in ('x', 'y', 'width', 'height')]
    if w <= 0 or h <= 0:
        raise ValueError('Rectangle width and height must be positive')
    return [[x,y],[x+w,y],[x+w,y+h],[x,y+h]], False


def translate_annotation(annotation, bound):
    """Map old canvas → old content − sampling shift → new canvas, no clipping."""
    points, polygon = _points(annotation)
    old_cw, old_ch, old_ox, old_oy, width, height = bound['old_layout']
    new_cw, new_ch, new_ox, new_oy, _, _ = bound['new_layout']
    dx, dy = bound['shift']
    output = []
    for x, y in points:
        ox, oy = x*old_cw-old_ox, y*old_ch-old_oy
        if min(ox, oy) < -1e-9 or ox > width+1e-9 or oy > height+1e-9:
            raise ValueError('Source annotation extends outside native content; clipping is forbidden')
        nx, ny = ox-dx, oy-dy
        if min(nx, ny) < -1e-9 or nx > width+1e-9 or ny > height+1e-9:
            raise ValueError('Migrated annotation extends outside native content; clipping is forbidden')
        output.append([(nx+new_ox)/new_cw, (ny+new_oy)/new_ch])
    result = deepcopy(annotation)
    xs, ys = list(zip(*output))
    result.update(x=min(xs), y=min(ys), width=max(xs)-min(xs), height=max(ys)-min(ys))
    result['points'] = output if polygon else []
    result['frame_index'] = bound['binding']['target_frame_index']
    result['elapsed_h'] = bound['new_frame']['elapsed_h']
    return result


def migrate_document(document, *, source_sha256, receipt_document, receipt_sha256,
                     source_manifests, target_manifests, source_hashes, target_hashes):
    """Pure detached transformation; caller binds file hashes and writes exclusively."""
    _hash(source_sha256, 'source decisions hash'); _hash(receipt_sha256, 'receipt document hash')
    if document.get('schema') != 2 or not isinstance(document.get('decisions'), dict):
        raise ValueError('Only Phase Studio schema 2 decisions are supported')
    if document.get('geometry_migration'):
        raise ValueError('Already migrated document: start from its original source for another revision')
    bound = _bind_receipts(receipt_document, source_manifests, target_manifests, source_hashes, target_hashes)
    result = deepcopy(document)
    records = []
    seen_runs, seen_ids = set(), set()
    for raw_run, decision in result['decisions'].items():
        run = str(raw_run).lower()
        if run in seen_runs or run not in source_manifests:
            raise ValueError(f'Missing or ambiguous run mapping: {run}')
        seen_runs.add(run)
        if decision.get('reference_guides'):
            raise ValueError('Reference guides have no supported per-frame geometry mapping')
        if decision.get('frame_choices') and any(k[0] == run and k[1] != b['binding']['target_frame_index'] for k,b in bound.items()):
            raise ValueError('Timeline change with frame_choices requires a separate review migration')
        for collection in ('annotations', 'phase_quiz_answers'):
            items = decision.get(collection) or []
            if not isinstance(items, list):
                raise ValueError(f'{collection} must be a list')
            migrated = []
            for original in items:
                identifier = original.get('id') or original.get('proposal_id')
                if not isinstance(identifier, str) or not identifier:
                    raise ValueError('Region needs id or proposal_id')
                identity = (run, collection, identifier)
                if identity in seen_ids:
                    raise ValueError(f'Duplicate annotation identity: {identity}')
                seen_ids.add(identity)
                index = _integer(original.get('frame_index'), 'annotation frame_index')
                key = (run, index)
                if key not in bound:
                    raise ValueError(f'Missing geometry receipt for annotated frame: {key}')
                entry = bound[key]
                elapsed = _number(original.get('elapsed_h'), 'annotation elapsed_h')
                if abs(elapsed-float(entry['old_frame']['elapsed_h'])) > 1e-9:
                    raise ValueError(f'Annotation time disagrees with source frame: {identity}')
                if original.get('training_sampling'):
                    raise ValueError('Pixel-sampling contracts must be rebuilt, never migrated as coordinates')
                new = translate_annotation(original, entry)
                original_geometry = {k:deepcopy(original[k]) for k in GEOMETRY_KEYS if k in original}
                record = {'run':run,'collection':collection,'id':identifier,'source_geometry':original_geometry,
                          'target_geometry':{k:deepcopy(new[k]) for k in GEOMETRY_KEYS if k in new},
                          'frame_binding':deepcopy(entry['binding'])}
                new['geometry_migration'] = {'format':FORMAT,'source_decisions_sha256':source_sha256,
                    'geometry_revision_id':receipt_document['geometry_revision_id'],
                    'source_geometry':original_geometry,'source_annotation_sha256':_sha_bytes(json.dumps(original,sort_keys=True,allow_nan=False).encode()),
                    'frame_receipt_sha256':_sha_bytes(json.dumps(entry['binding'],sort_keys=True,allow_nan=False).encode()),
                    'candidate_attribution_applies_to_original_prediction_only':True}
                migrated.append(new); records.append(record)
            if collection in decision:
                decision[collection] = migrated
    result['geometry_migration'] = {
        'format':FORMAT,'created_utc':datetime.now(timezone.utc).isoformat(),
        'geometry_revision_id':receipt_document['geometry_revision_id'],
        'source_decisions_sha256':source_sha256,'receipt_document_sha256':receipt_sha256,
        'source_manifest_sha256':dict(source_hashes),'target_manifest_sha256':dict(target_hashes),
        'coordinates':'old canvas -> old content minus sampling displacement -> target canvas; continuous image edges; no clipping',
        'regions_migrated':len(records),'records':records,
        'original_document':deepcopy(document),
        'activation':'Detached evidence copy only. Not registered with Studio and not a review of new predictions.',
        'requires_review_before_training':True,'training_performed':False,
    }
    return result


def migrate_copy(source_decisions_path, *, source_manifest_paths: Mapping[str, Path],
                 target_manifest_paths: Mapping[str, Path], receipt_path, output_path,
                 expected_source_sha256=None):
    source, output, receipt_path = Path(source_decisions_path), Path(output_path), Path(receipt_path)
    for manifests in (source_manifest_paths, target_manifest_paths):
        if len({str(k).lower() for k in manifests}) != len(manifests):
            raise ValueError('Ambiguous case-insensitive manifest run keys')
    if {k.lower() for k in source_manifest_paths} != {k.lower() for k in target_manifest_paths}:
        raise ValueError('Source and target manifest run sets must match')
    paths=[source,receipt_path,*source_manifest_paths.values(),*target_manifest_paths.values()]
    if output.resolve() in [Path(p).resolve() for p in paths] or output.exists():
        raise ValueError('Output must be a new detached path, never an input or existing file')
    loaded={str(Path(p).resolve()):_read(p) for p in paths}
    def get(p): return loaded[str(Path(p).resolve())]
    document, source_hash=get(source)
    if expected_source_sha256 is not None and source_hash != expected_source_sha256:
        raise ValueError('Source decisions changed since selection')
    receipts, receipt_hash=get(receipt_path)
    result=migrate_document(document,source_sha256=source_hash,receipt_document=receipts,receipt_sha256=receipt_hash,
        source_manifests={k.lower():get(p)[0] for k,p in source_manifest_paths.items()},
        target_manifests={k.lower():get(p)[0] for k,p in target_manifest_paths.items()},
        source_hashes={k.lower():get(p)[1] for k,p in source_manifest_paths.items()},
        target_hashes={k.lower():get(p)[1] for k,p in target_manifest_paths.items()})
    result['geometry_migration']['source_decisions_path']=str(source.resolve())
    result['geometry_migration']['receipt_document_path']=str(receipt_path.resolve())
    result['geometry_migration']['source_manifest_paths']={k.lower():str(Path(p).resolve()) for k,p in source_manifest_paths.items()}
    result['geometry_migration']['target_manifest_paths']={k.lower():str(Path(p).resolve()) for k,p in target_manifest_paths.items()}
    # Build the entire result before writing anything. User editing can continue;
    # a changed input invalidates this copy operation rather than being mixed in.
    encoded=json.dumps(result,indent=2,allow_nan=False).encode('utf-8')
    for p,(payload,expected_hash) in loaded.items():
        if _sha_bytes(Path(p).read_bytes()) != expected_hash:
            raise ValueError(f'Input changed during migration: {p}')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('xb') as stream: stream.write(encoded)
    return {'output_path':str(output.resolve()),'output_sha256':_sha_bytes(encoded),
            'source_decisions_sha256':source_hash,'regions_migrated':result['geometry_migration']['regions_migrated'],
            'training_performed':False,'studio_registration_performed':False}


def _assignments(values):
    result={}
    for value in values:
        run,separator,path=value.partition('=')
        run=run.lower()
        if not separator or not run or run in result:
            raise ValueError('Use one unique run=manifest.json assignment per run')
        result[run]=Path(path)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--source-manifest',action='append',required=True,metavar='RUN=PATH')
    parser.add_argument('--target-manifest',action='append',required=True,metavar='RUN=PATH')
    parser.add_argument('--receipts',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--expected-source-sha256')
    args=parser.parse_args()
    print(json.dumps(migrate_copy(args.source,source_manifest_paths=_assignments(args.source_manifest),
        target_manifest_paths=_assignments(args.target_manifest),receipt_path=args.receipts,
        output_path=args.output,expected_source_sha256=args.expected_source_sha256),indent=2))


if __name__=='__main__': main()
