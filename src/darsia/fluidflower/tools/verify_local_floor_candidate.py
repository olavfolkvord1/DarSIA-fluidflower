"""Compare a local-floor revision to its parent on identical archived frames."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from darsia.fluidflower.phase_runtime import load_physics_profile, load_run_context, phase_physics_sequence
from darsia.fluidflower.phase_validation import _source_jobs
from darsia.fluidflower.project import sha256_file


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def masks(root, job, frame, crop, raw=False):
    if raw:
        fingerprint = read(root / 'validation/validation_binding.json')['model_sha256']
        assets = root / 'validation/raw_optical_assets'
    else:
        fingerprint = job['params_sha256']
        assets = Path(job['phase_map_output_dir'])
    path = assets / 'native_phase_maps' / job['run'] / f'A_{fingerprint[:16]}' / f'frame_{frame:03d}.png'
    x, y, w, h = crop
    with Image.open(path) as image:
        data = np.asarray(image)[y:y+h, x:x+w]
        c = data[..., 3] > 0
        return c, c & (data[..., 0] == 242)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    parent_jobs = {j['run']: j for j in _source_jobs(args.parent / 'validation')}
    profile = load_physics_profile(args.model)
    report = {'scope': 'Paired support regression; not independent physical truth', 'runs': {}}
    for job in _source_jobs(args.model / 'validation'):
        run = job['run']
        entry = profile['runs'][f"{job['dataset']}/{run}"]
        ctx, _ = load_run_context(args.model, profile, job['dataset'], run,
                                  native_manifest_sha256=entry['native_manifest_sha256'])
        seq = phase_physics_sequence(ctx, profile)
        meta = read(Path(job['review_output_dir']) / 'native_frames' / run / 'manifest.json')
        old_meta = read(Path(parent_jobs[run]['review_output_dir']) / 'native_frames' / run / 'manifest.json')
        assert meta['content_box_xywh'] == old_meta['content_box_xywh']
        assert meta['frame_sources'] == old_meta['frame_sources']
        rows = []
        for frame, item in enumerate(meta['frame_sources']):
            c, g = masks(args.model, job, frame, meta['content_box_xywh'])
            old_c, old_g = masks(args.parent, parent_jobs[run], frame, meta['content_box_xywh'])
            raw_c, raw_g = masks(args.model, job, frame, meta['content_box_xywh'], raw=True)
            parent_c, parent_g = masks(args.parent, parent_jobs[run], frame, meta['content_box_xywh'], raw=True)
            assert np.array_equal(raw_c, parent_c) and np.array_equal(raw_g, parent_g), (run, frame, 'optics changed')
            assert np.array_equal(c, old_c), (run, frame, 'CO2 presence changed')
            assert not np.any(g & ~old_g), (run, frame, 'new gas created')
            replay, _ = seq.apply({'co2_presence_probability': raw_c.astype(np.float32),
                'gas_probability_conditional_on_co2': raw_g.astype(np.float32)}, item['elapsed_h'])
            assert np.array_equal(g, (replay['co2_presence_probability'] >= .5) &
                                    (replay['gas_probability_conditional_on_co2'] >= .6)), (run, frame, 'replay mismatch')
            protected = np.zeros_like(seq.valid)
            violations = np.zeros_like(seq.valid)
            for index, (stage, history) in enumerate(zip(ctx.injection_stages, seq.history), 1):
                owned = seq.owner == index
                protected |= owned & seq.near(history, seq.settings.registration_tolerance_m)
                if item['elapsed_h'] >= stage['end_h']:
                    violations |= owned & ~seq.local_envelope(history)
            removed = old_g & ~g
            assert not np.any(removed & seq.full(protected)), (run, frame, 'historic gas lost')
            assert not np.any(g & seq.full(violations)), (run, frame, 'local floor violated')
            row = {'frame': frame, 'time_h': item['elapsed_h'], 'old_gas_pixels': int(old_g.sum()),
                   'new_gas_pixels': int(g.sum()), 'reclassified_to_aqueous': int(removed.sum())}
            rows.append(row)
            if run == 'ac61' and frame in (14, 20):
                colors = np.array([[0, 0, 0], [49, 217, 160], [242, 76, 157]], np.uint8)
                fig, axes = plt.subplots(1, 3, figsize=(17, 4.5), constrained_layout=True)
                for ax, mask, title in zip(axes[:2], (old_g, g), ('V12', 'V12-F1: local floor')):
                    labels = np.where(seq.small(c), np.where(seq.small(mask), 2, 1), 0)
                    ax.imshow(colors[labels])
                    ax.set_title(title)
                axes[2].imshow(seq.small(removed), cmap='gray')
                axes[2].set_title('Gas reclassified to aqueous')
                for ax in axes:
                    ax.axis('off')
                fig.suptitle(f'AC61 at {item["elapsed_h"]:.2f} h; identical optics, different phase support')
                fig.savefig(args.output / f'ac61_frame_{frame:03d}.png', dpi=150)
                plt.close(fig)
            print(json.dumps({'run': run, **row}), flush=True)
        report['runs'][run] = rows
    provenance = read(args.model / 'revision_provenance.json')
    for path, digest in provenance['protected_sources'].items():
        assert sha256_file(Path(path)) == digest, path
    report.update(all_frames_same_optics=True, co2_presence_unchanged=True,
        no_additional_loss_inside_historic_gas_support=True, parent_unchanged=True,
        no_gas_below_observed_local_floor=True)
    with (args.output / 'report.json').open('x') as stream:
        json.dump(report, stream, indent=2)


if __name__ == '__main__':
    main()
