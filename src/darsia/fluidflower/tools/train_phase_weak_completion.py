"""Bounded aqueous completion experiment; never mutates Studio models or labels.

Learning uses annotation semantics. Retention of the old map is an inference
invariant, not a claim that the old map is ground truth.
"""
from __future__ import annotations

from darsia.fluidflower.tools._paths import project_path, source_path, process_directory, series_root, series_path, tool_path, local_cache_root, package_root

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time

import cv2
import numpy as np

REPO = series_root("ffac")
None
from darsia.fluidflower.tools import phase_spatial_context_pilot as previous
from darsia.fluidflower.tools.phase_spatial_pilot_evidence import CAMPAIGN, load_frame, read, sha, write, verify
from darsia.fluidflower.tools.phase_model_comparison_unet import _build_model

SOURCE = CAMPAIGN / 'context_corrector_balanced'
ROOT = CAMPAIGN / 'weak_completion'
SEED = 20260914
PATCH = 128
SUPERVISED = {'aqueous', 'water', 'gas', 'aqueous_absent_region',
              'co2_absent_region', 'aqueous_present_region',
              'mixed_gas_aqueous', 'mixed_aqueous_water'}


def aqueous_loss(q, weights, phase_class):
    """Latent aqueous supervision, before native retention/exclusion is applied."""
    import torch
    if phase_class not in SUPERVISED:
        raise ValueError('No independent aqueous target: ' + phase_class)
    dense = lambda p: previous._weighted_mean(-torch.log(p.clamp_min(1e-7)), weights)
    if phase_class == 'aqueous':
        return dense(q)
    if phase_class in {'water', 'gas', 'aqueous_absent_region', 'co2_absent_region'}:
        return dense(1 - q)
    if phase_class in {'aqueous_present_region', 'mixed_gas_aqueous'}:
        return .35 * previous._mil(q, weights)
    return .35 * (previous._mil(q, weights) + previous._mil(1 - q, weights))


def inputs(path, contextual):
    with np.load(path) as a:
        rgb = a['rgb'].astype(np.float32).transpose(2, 0, 1) / 255
        x = np.concatenate([rgb, a['raw'][None].astype(np.float32),
                            a['co2'][None].astype(np.float32), a['gas'][None].astype(np.float32)])
        domain = a['domain'].astype(np.float32)
    if contextual:
        # Three millimetre sigma on the fixed one millimetre working grid.
        delta = np.stack([cv2.GaussianBlur(rgb[i] - rgb[i + 3], (0, 0), 3,
                                        borderType=cv2.BORDER_REFLECT_101) for i in range(3)])
        x = np.concatenate([x, delta])
    return x, domain


def base_probability(x):
    return .2 + .6 * (x[:, 7] - x[:, 8]).clamp(0, 1)


def prediction(model, x):
    import torch
    return torch.sigmoid(torch.logit(base_probability(x)) + model(x)[:, 0])


def network(torch, contextual):
    model = _build_model(torch, base_channels=8)
    model.encoder1.layers[0] = torch.nn.Conv2d(12 if contextual else 9, 8, 3, padding=1)
    model.heads = torch.nn.Conv2d(8, 1, 1)
    torch.nn.init.zeros_(model.heads.weight)
    torch.nn.init.zeros_(model.heads.bias)
    return model


def prepared_paths(manifest):
    receipts = {f['frame_id']: f for f in read(SOURCE / 'preparation.json')['frames']}
    paths = {}
    for entry in manifest['frames']:
        key = entry['frame_id']
        if key in receipts:
            record = receipts[key]
            verify(record['path'], record['sha256'])
            paths[key] = Path(record['path'])
            continue
        # The unlabelled AC44 transfer frame was not prepared for the old fit.
        f = load_frame(manifest, key)
        nh, nw = f['domain'].shape
        dy, dx = f['spacing_m']
        target = (round(nw * dx / .001), round(nh * dy / .001))
        resize = lambda value: cv2.resize(value, target, interpolation=cv2.INTER_AREA)
        path = ROOT / 'prepared' / (key + '.npz')
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path,
            rgb=np.concatenate([resize(f[k]) for k in ('current_rgb', 'baseline_rgb')], axis=-1),
            raw=resize(f['raw_co2']).astype(np.float16),
            co2=resize((f['saved_labels'] > 0).astype(np.float32)).astype(np.float16),
            gas=resize((f['saved_labels'] == 2).astype(np.float32)).astype(np.float16),
            domain=resize(f['domain'].astype(np.float32)).astype(np.float16))
        paths[key] = path
        del f
    return paths


def fit(data, paths, contextual, output):
    import torch
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    entries = [f for f in data['frames'] if f['run'] != 'run18' and f['regions']]
    arrays = {}
    for entry in entries:
        x, d = inputs(paths[entry['frame_id']], contextual)
        arrays[entry['frame_id']] = (torch.from_numpy(x).cuda(), torch.from_numpy(d).cuda())
    bags = []
    ignored = Counter()
    for entry in entries:
        for record in entry['regions']:
            if record['phase_class'] not in SUPERVISED:
                ignored[record['phase_class']] += 1
                continue
            bags.append(previous.make_bag(entry, record, arrays[entry['frame_id']][0].shape[-2:]))
    strata = Counter((b['record']['dataset'], b['record']['run'], b['record']['phase_class']) for b in bags)
    classes = Counter(k[:2] for k in strata)
    for b in bags:
        key = (b['record']['dataset'], b['record']['run'], b['record']['phase_class'])
        b['weight'] = len(bags) / (len(classes) * classes[key[:2]] * strata[key])
    model = network(torch, contextual).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-4)
    rng = np.random.default_rng(SEED)
    rows = []
    start = time.perf_counter()
    steps = 0
    for epoch in range(8):
        model.train()
        objectives = []
        for idx in rng.permutation(len(bags)):
            bag = bags[int(idx)]
            x, domain, mask = previous._bag_arrays(torch, bag, arrays)
            optimizer.zero_grad(set_to_none=True)
            q = prediction(model, x)
            supervision = aqueous_loss(q, mask, bag['record']['phase_class']) * bag['weight']
            tether = previous._weighted_mean((q - base_probability(x)).square(), domain)
            loss = supervision + .02 * tether
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            steps += 1
            objectives.append(float(loss.detach()))
            if time.perf_counter() - start > 480:
                raise RuntimeError('Fixed fit exceeded eight minute budget; incomplete candidate is invalid')
        row = dict(epoch=epoch + 1, steps=steps, objective=float(np.mean(objectives)), seconds=time.perf_counter()-start)
        rows.append(row)
        print(json.dumps(dict(variant=output.name, **row)), flush=True)
    checkpoint = output / 'aqueous_completion.pt'
    torch.save(dict(state_dict=model.state_dict(), contextual=contextual, seed=SEED), checkpoint)
    report = dict(complete=True, epochs=8, steps=steps, training_seconds=time.perf_counter()-start,
        parameters=sum(p.numel() for p in model.parameters()), trained_regions=len(bags),
        ignored_no_aqueous_supervision=dict(ignored), strata={str(k): v for k, v in strata.items()},
        history=rows, checkpoint_sha256=sha(checkpoint), held_run_labels_used=False)
    write(output / 'fit_report.json', report)
    del arrays, optimizer
    torch.cuda.empty_cache()
    return model


def infer(model, x):
    import torch
    _, h, w = x.shape
    starts = lambda n: sorted(set(range(0, max(n-PATCH, 0)+1, PATCH-32)) | {max(0, n-PATCH)})
    locations = [(y, z) for y in starts(h) for z in starts(w)]
    numerator = np.zeros((h, w), np.float32)
    mass = np.zeros((h, w), np.float32)
    ramp = np.maximum(np.hanning(PATCH).astype(np.float32), .05)
    window = ramp[:, None] * ramp[None, :]
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(locations), 8):
            group = locations[start:start+8]
            batch = torch.stack([x[:, y:y+PATCH, z:z+PATCH] for y, z in group])
            values = prediction(model, batch).cpu().numpy()
            for (y, z), q in zip(group, values):
                numerator[y:y+PATCH, z:z+PATCH] += q * window
                mass[y:y+PATCH, z:z+PATCH] += window
    if not np.all(mass > 0):
        raise ValueError('Incomplete frame inference')
    return numerator / mass


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=['aqueous_only', 'reference_context', 'all'], default='all')
    args = parser.parse_args()
    cv2.setNumThreads(4)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest = SOURCE / 'evidence_manifest.json'
    data = read(manifest)
    ROOT.mkdir(exist_ok=True)
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(), source_manifest=str(manifest),
        source_manifest_sha256=sha(manifest), source_sha256=sha(__file__), seed=SEED,
        variants=['aqueous_only', 'reference_context'], epochs=8, learning_rate=.001,
        prior_weight=.02, spatial_shrinkage_loss=False, working_spacing_m=.001,
        input_difference='Reference-context variant adds three Gaussian-smoothed current-minus-baseline channels, sigma3mm. Otherwise identical fits.',
        training='Same non-RUN18 balanced cases. Only classes with independent aqueous supervision enter class/run weighting. Presence means whole-bag MIL, never a dense box.',
        inference='Native map union: preserve all old aqueous/gas, add aqueous only on old-water domain pixels with interpolated learned q>=0.5.',
        no_old_map_as_training_truth=True, no_temporal_input=True, no_label_overrides=True,
        evaluation='Same original12 frames. RUN18 excluded from this fit, but previously used for upstream fit and repeated development feedback. Other labelled frames are training controls; AC44 unlabelled.',
        acceptance='No promotion from retention invariants. Assess additions, negative regressions, full-frame expansion and continuity of fixed user-shown weak regions. No held-run checkpoint selection.',
        production_modified=False)
    pp = ROOT / 'protocol.json'
    if pp.exists():
        old = read(pp)
        for key in ('source_manifest_sha256', 'source_sha256'):
            if old[key] != protocol[key]:
                raise ValueError('Frozen completion experiment source changed')
    else:
        write(pp, protocol)
        snapshot = ROOT / 'source_snapshot'
        snapshot.mkdir()
        for name in ('scripts/train_phase_weak_completion.py', 'scripts/phase_spatial_context_pilot.py',
                     'scripts/phase_spatial_pilot_evidence.py', 'scripts/phase_model_comparison_unet.py'):
            shutil.copyfile(source_path(name), snapshot / Path(name).name)
    paths = prepared_paths(data)
    write(ROOT / 'prepared_sources.json', {key: dict(path=str(path), sha256=sha(path)) for key, path in paths.items()})
    evaluation_ids = {f['frame_id'] for f in read(CAMPAIGN / 'evidence_manifest.json')['frames']}
    for name in protocol['variants']:
        if args.variant not in ('all', name):
            continue
        output = ROOT / name
        if output.exists():
            raise FileExistsError('Do not overwrite a completion experiment: ' + str(output))
        output.mkdir()
        contextual = name == 'reference_context'
        model = fit(data, paths, contextual, output)
        receipts = []
        for entry in data['frames']:
            key = entry['frame_id']
            if key not in evaluation_ids:
                continue
            x, domain = inputs(paths[key], contextual)
            q = infer(model, torch.from_numpy(x).cuda())
            path = output / (key + '_aqueous_probability.npy')
            np.save(path, q)
            receipts.append(dict(frame_id=key, path=str(path), sha256=sha(path), shape=list(q.shape)))
            print(json.dumps(dict(inferred=key, variant=name)), flush=True)
        write(output / 'prediction_receipts.json', dict(complete=True, frames=receipts))
        del model
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
