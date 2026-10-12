"""Cached exact-image local-support preview, with no repeated phase inference."""
from __future__ import annotations

from darsia.fluidflower.series_paths import local_cache_root
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def resolve(root, run, candidate, index, layer='all', *, reference_mode=False, reference=None):
    from darsia.fluidflower.tools import calibration_pairwise_review as review
    from darsia.fluidflower.tools.phase_reference_maps import build_plan, resolve_reference_overlay
    from darsia.fluidflower.tools.phase_mass_overnight_data import crop_rgb, decode_labels, sha, write
    from darsia.fluidflower.tools.phase_injection_references import resolve_reference_asset
    from darsia.fluidflower import phase_runtime
    from darsia.fluidflower import phase_local_aqueous_support as local
    from darsia.fluidflower.phase_validation import PHASE_FILL_COLORS
    plan = build_plan(None, root, run, candidate)
    native = json.loads(Path(plan['native_manifest']['path']).read_text(encoding='utf-8-sig'))
    if reference_mode:
        photo = resolve_reference_asset(reference, root, run) if reference is not None else Path(plan['baseline']['path'])
        original = resolve_reference_overlay(reference, root, run, candidate, 'A', 'all')
        elapsed = float(reference['elapsed_h']) if reference is not None else -1.
    else:
        photo = review._native_frame_path(root, run, index)
        original = review._native_phase_map_path(root, candidate, 'A', index)
        elapsed = float(native['frame_sources'][index]['elapsed_h'])
    profile = phase_runtime.load_physics_profile(plan['model_root'])
    if profile and profile.get('unified_phase_decision'):
        # The selected method already owns reference evidence and the contour.
        # Old saved toggle preferences must never append a second decision.
        if reference_mode:
            return resolve_reference_overlay(reference, root, run, candidate, 'A', layer)
        return review.two_phase_overlay_path(original, layer)
    settings = local.Settings()
    identity = dict(format='local-reference-support-preview-v2', phase_sha256=sha(original),
        photo_sha256=sha(photo), baseline_sha256=plan['baseline']['sha256'],
        inherited_runtime=plan['physics_runtime'], settings=asdict(settings),
        implementation_sha256=sha(local.__file__), frame_time_h=elapsed, model_negative_veto=False)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    folder = local_cache_root() / 'phase_local_support_preview'/key
    folder.parent.mkdir(parents=True, exist_ok=True)
    target = folder/('all.png' if layer == 'all' else layer+'.png')
    with review._exclusive_path_lock(folder.with_suffix('.lock'), timeout_s=1200, stale_after_s=3600):
        if target.is_file() and (folder/'receipt.json').is_file(): return target
        folder.mkdir(parents=True, exist_ok=True)
        context, _ = phase_runtime.load_run_context(plan['model_root'], profile, plan['dataset'], run,
            native_manifest_sha256=plan['native_manifest']['sha256'])
        box = native['content_box_xywh']; labels = decode_labels(original, box)
        added = np.zeros(context.domain.shape, bool)
        report = dict(reason='pre_injection', added_pixels=0)
        if elapsed > 0:
            current = np.rint(crop_rgb(photo, box)*255).astype(np.uint8)
            baseline = np.rint(crop_rgb(plan['baseline']['path'], box)*255).astype(np.uint8)
            spacing = (abs(float(context.y_m[1]-context.y_m[0])), abs(float(context.x_m[1]-context.x_m[0])))
            added, report = local.regional_support(current, baseline, labels, context.domain,
                                                   context.facies, spacing, settings)
        labels[added] = 1
        x, y, w, h = box; cw, ch = native['canvas_wh']
        for mode in ('all', 'aqueous', 'gas'):
            rgba = np.zeros((ch, cw, 4), np.uint8)
            for code, name in ((1, 'aqueous'), (2, 'gas')):
                if mode in ('all', name): rgba[y:y+h, x:x+w][labels == code] = PHASE_FILL_COLORS[name]
            Image.fromarray(rgba).save(folder/(mode+'.png'))
        write(folder/'receipt.json', dict(identity=identity, report=report,
            interpretation='Local baseline search with unified continuous boundary. Existing phases retained; earlier annotations are not a hard veto. This changes categorical phase support, not just display smoothing.',
            original_phase=str(original), photo=str(photo), model_root=plan['model_root']))
    return target
