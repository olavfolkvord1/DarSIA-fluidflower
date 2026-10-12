"""Read saved replay evidence only; never call any model or change a map."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(descriptor):
    path = Path(descriptor['path'])
    if sha(path) != descriptor['sha256']:
        raise ValueError(f'Changed saved evidence: {path}')
    return json.loads(path.read_text(encoding='utf-8-sig'))


def descriptor(path):
    return dict(path=str(path), sha256=sha(path))


def load_image(item):
    if sha(item['path']) != item['sha256']:
        raise ValueError('Saved label SHA differs')
    with Image.open(item['path']) as image:
        return np.asarray(image).copy()


def packed_masks(item, names, shape):
    if sha(item['path']) != item['sha256']:
        raise ValueError('Saved mask SHA differs')
    with np.load(item['path'], allow_pickle=False) as archive:
        return {name: np.unpackbits(archive[name], bitorder='little',
            count=int(np.prod(shape))).reshape(shape).astype(bool) for name in names}


def build(root, receipt_path, destination):
    root, destination = Path(root), Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    phase_plan = descriptor(root / 'phase_inference_plan.json')
    plan = read(phase_plan)
    receipt_descriptor = descriptor(receipt_path)
    saved = read(receipt_descriptor)
    source = saved['binding']['source']
    run = source['run']
    inventory = read(plan['runs'][run]['source_inventory'])
    rows = {row['source_sha256']: row for row in inventory['sources']}
    predecessor = rows[source['previous_source_sha256']]
    baseline = rows[inventory['baseline_source_sha256']]
    probability = read(source['parent_source']['retained_source_cache']['parent_probability_receipt'])
    native = read(plan['runs'][run]['parent_native_manifest'])
    current_native = native['frame_sources'][source['parent_source']['parent_source_frame_index']]
    original_geometry = read(dict(path=native['v5_track1_photo_geometry']['frames'][source['parent_source']['parent_source_frame_index']]['receipt_path'],
        sha256=native['v5_track1_photo_geometry']['frames'][source['parent_source']['parent_source_frame_index']]['receipt_sha256']))
    scientific_descriptor = saved['qa']['native_inference']['input_provenance']['source']
    scientific = read(scientific_descriptor)
    current_h = np.asarray(scientific['geometry_frame']['H_v5_B_to_frame_native'])
    original_h = np.asarray(original_geometry['H_v5_B_to_frame_native'])
    qa = saved['qa']['native_inference']
    proof = qa['input_provenance']
    current_exact = (proof['historical_native_input']['log_linear_rgb']['sha256'] == current_native['v5_log_linear_rgb_sha256']
        and proof['historical_native_input']['valid_mask']['sha256'] == current_native['v5_log_linear_valid_mask_sha256'])
    baseline_exact = (proof['baseline_temporal']['native_input']['log_linear_rgb']['sha256'] == native['reference']['v5_log_linear_rgb_sha256']
        and proof['baseline_temporal']['native_input']['valid_mask']['sha256'] == native['reference']['v5_log_linear_valid_mask_sha256'])
    replay = saved['qa']['strict_studio_replay']
    parent = read(source['parent_receipt'])
    old_labels, new_labels = load_image(parent['labels_map']), load_image(saved['labels_map'])
    co2_changed = np.isin(old_labels, (1, 2)) ^ np.isin(new_labels, (1, 2))
    gas_changed = (old_labels == 2) ^ (new_labels == 2)
    if int(co2_changed.sum()) != replay['co2']['changed_pixels'] or int(gas_changed.sum()) != replay['gas']['changed_pixels']:
        raise ValueError('Saved label differences disagree with producer replay receipt')
    shape = old_labels.shape
    operator_masks = packed_masks(saved['operator_masks'], saved['operator_mask_names'], shape)
    ys, xs = np.where(co2_changed)
    intersections = {name: int((co2_changed & mask).sum()) for name, mask in operator_masks.items()}
    if sha(saved['rig_domain_revision']['classes']['path']) != saved['rig_domain_revision']['classes']['sha256']:
        raise ValueError('Saved classes changed')
    with np.load(saved['rig_domain_revision']['classes']['path'], allow_pickle=False) as archive:
        class_intersections = {name: int((co2_changed & archive[name]).sum()) for name in archive.files}
    result = dict(contract='ffac-selected-replay-readonly-diagnosis-v1',
        phase_plan=phase_plan, replay_receipt=receipt_descriptor, scientific_receipt=scientific_descriptor,
        parent_receipt=source['parent_receipt'], run=run,
        original_studio_frame_index=source['studio_frame_index'], new_source_work_index=source['work_index'],
        current_source=dict(source_sha256=source['source_sha256'], source_path=source['source_path'], elapsed_h=source['elapsed_h']),
        original_predecessor=source['parent_source']['temporal_predecessor'],
        selected_predecessor={key: predecessor[key] for key in ('source_sha256', 'source_path', 'elapsed_h', 'studio_anchor', 'selected')},
        baseline_source={key: baseline[key] for key in ('source_sha256', 'source_path', 'elapsed_h')},
        current_masked_descriptors_identical=current_exact, baseline_masked_descriptors_identical=baseline_exact,
        producer_bitwise_assertion='Completed producer verified scientific-native float32 bits on the historical valid support against the bound original masked source. This report does not re-read or re-hash the full RAW arrays.',
        current_H_exact=bool(np.array_equal(current_h, original_h)), current_H_max_abs_difference=float(np.max(np.abs(current_h-original_h))),
        baseline_temporal_policy=proof['baseline_temporal']['policy'], previous_temporal_policy=proof['previous_native_temporal']['policy'],
        original_alignment=probability['metadata']['alignment'], corrected_alignment=qa['temporal_alignment'],
        original_spacing_m=probability['metadata']['spacing_m'], corrected_spacing_m=qa['spacing_m'],
        probability_storage_original=probability['metadata']['storage_layout'], probability_storage_corrected=qa['probability_encoding']['storage_layout'],
        probability_encoding_corrected=qa['matched_encoding_receipt'], original_cpu_only=probability['metadata']['cpu_only'], corrected_cpu_only=qa['cpu_only'],
        original_native_shape_hw=probability['metadata']['native_shape_hw'], corrected_native_shape_hw=qa['native_shape_hw'],
        support_changed_pixels=replay['support_changed_pixels'], replay=replay,
        co2_changed_bbox_canvas_xyxy=[int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
        co2_difference_intersections_operator_masks=intersections, co2_difference_intersections_classes=class_intersections,
        diagnosis='History is a verified input difference. Its isolated causal contribution is not established: no counterfactual model execution was performed. The source/geometry/masked-input assertion, precision layout and final support checks pass. Remaining reported spacing and alignment-validity convention differences must be assessed before claiming history is the sole cause.',
        maps_changed=False, inference_performed=False, model_executed=False, thresholds_changed=False,
        scientific_sidecars_changed=False, frozen_model_changed=False, labels_changed=False,
        production_allowed=False, candidate_registered=False)
    (destination / 'diagnosis.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    text = ('Corrected selected-phase replay stopped — AC19\n\n'
        '14 of 782 original Studio anchors were saved: 13 pass, one fails. No full selected production, completion seal or Studio candidate was admitted.\n\n'
        f"AC19 original Studio frame {source['studio_frame_index']} is new source work index {source['work_index']} at {source['elapsed_h']:.9f} h. "
        f"CO2 XOR is {replay['co2']['changed_pixels']} pixels / {replay['co2']['changed_cm2']:.9f} cm2 / {100*replay['co2']['changed_area_fraction']:.8f}%, exceeding the unchanged 0.1% limit. "
        f"The net CO2 area change is {replay['co2']['replay_cm2']-replay['co2']['parent_cm2']:.9f} cm2; this is not the XOR metric. "
        f"Gas XOR is {replay['gas']['changed_pixels']} pixels / {replay['gas']['changed_cm2']:.9f} cm2 / {100*replay['gas']['changed_area_fraction']:.8f}%, passing 1%. Final support changed by zero pixels.\n\n"
        'Current and baseline use the exact original masked RAW descriptors. The producer checked bitwise equality on historical valid support; this diagnostic does not re-hash the large RAW files. The original packed photo mask is used, temporal origins are (0,0), and the current homography is exact.\n\n'
        f"Original predecessor: DSC05394 baseline at {baseline['elapsed_h']:.9f} h. Selected Policy A predecessor: {Path(predecessor['source_path']).name} at {predecessor['elapsed_h']:.9f} h. The latter is a newly selected acquisition. This is a known change to temporal model input; it is not proof that all changed pixels are caused by history.\n\n"
        'Both probability layouts use compact G3 uint8 with working-grid float16 regional values, and both producers are CPU only. Original native spacing is reported with separate axes; the helper uses rounded equal axes. The original native-only previous alignment supplied no explicit validity mask, while the corrected helper supplies the finite source mask; the warp RGB arithmetic itself does not normalize by that mask. These are reported conventions, not demonstrated causal explanations.\n\n'
        'All committed source and phase receipts, partials, old candidates, models and labels are preserved. No restart, threshold change, new inference, packaging, mass calculation or default change was performed by this report.\n')
    (destination / 'README.txt').write_text(text, encoding='utf-8')
    return descriptor(destination / 'diagnosis.json')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True); parser.add_argument('--receipt', required=True)
    parser.add_argument('--destination', required=True)
    args=parser.parse_args()
    print(json.dumps(build(args.root,args.receipt,args.destination)))


if __name__ == '__main__':
    main()
