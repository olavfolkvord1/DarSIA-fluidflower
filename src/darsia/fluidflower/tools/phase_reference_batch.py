"""Resolve all display layers from one verified reference plan.

The prediction/replay implementation and its cache identity are unchanged.
Only batch orchestration avoids rebuilding the same plan for each layer.
"""
from pathlib import Path

from darsia.fluidflower.tools import phase_reference_maps as maps


class ReferenceInferenceRequired(RuntimeError):
    """A CPU finalizer must return this work to the exclusive GPU stage."""


def resolve_layers(reference, review_root, run, candidate, layers, *, progress=None, allow_inference=True):
    requested = set(layers)
    if not requested or not requested <= maps.LAYERS:
        raise ValueError('Unknown or empty reference layers')
    with maps._RESOLVE_LOCK:
        plan = maps.build_plan(reference, Path(review_root), run, candidate)
        first = 'all' if 'all' in requested else sorted(requested)[0]
        if maps._cached(plan, first) is None:
            single_pass = (Path(plan['binding']['path']).parent / 'single_pass_references'
                           / plan['run'] / plan['reference']['id'] / 'single_pass.json')
            if plan.get('reviewed_layers', {}).get('all') or single_pass.is_file():
                maps.compute_request(plan)
            elif not allow_inference:
                raise ReferenceInferenceRequired(f"Reference requires exclusive inference: {run}")
            else:
                maps._resolve_fresh(plan, first, progress=progress)
        receipt = maps._read(maps.cache_root() / plan['key'] / 'receipt.json')
        if receipt.get('key') != plan['key']:
            raise ValueError('Reference map receipt identity differs')
        result = {}
        for layer in sorted(requested):
            item = receipt.get('layers', {}).get(layer)
            if item is None:
                raise ValueError(f'Reference phase layer is unavailable: {run}/{layer}')
            path = Path(item['path'])
            if not path.is_file() or not path.stat().st_size:
                raise ValueError(f'Reference phase layer is unavailable: {run}/{layer}')
            if maps._sha(path) != item['sha256']:
                raise ValueError('Reference map asset differs from its receipt')
            result[layer] = dict(path=str(path.resolve()), sha256=item['sha256'])
        maps._check_assets(plan)
        return result
