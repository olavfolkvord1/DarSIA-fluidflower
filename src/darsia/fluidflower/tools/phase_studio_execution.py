"""CPU-first portable inference for ordinary Studio materialization."""
from __future__ import annotations

import time
import numpy as np


class CpuPredictor:
    def __init__(self, model, baseline, report, threads=1, *, optical_cache_root=None,
                 optical_cache_max_bytes=None, verify_numerics=False):
        from darsia.fluidflower.tools.phase_optical_cache import create_cache
        from darsia.fluidflower.phase_training.radiometry import model_profile
        self.model, self.baseline = model, baseline
        self.profile = model_profile(model)
        self.threads, self._tiles = threads, None
        self.verify_numerics = bool(verify_numerics)
        self.cache, reason = create_cache(model, baseline, "cpu", root=optical_cache_root,
                                         max_bytes=optical_cache_max_bytes)
        self.checked = False
        self.fallback = False
        self.report = dict(backend="cpu-tiles-checked" if self.verify_numerics else "cpu-tiles",
                           verify_numerics=self.verify_numerics, frames=[], threads=threads)
        if self.cache is not None:
            self.report["optical_cache"] = self.cache.stats
        elif reason:
            self.report["optical_cache_unavailable"] = reason
        report.setdefault("runs", []).append(self.report)

    @property
    def tiles(self):
        if self._tiles is None:
            from darsia.fluidflower.tools.phase_optical_exact_tiles import ExactTilePredictor
            self._tiles = ExactTilePredictor(self.model, self.baseline, threads=self.threads,
                max_window_px=256, radiometry_profile=self.profile)
        return self._tiles

    def __call__(self, current, previous):
        from darsia.fluidflower.tools.phase_optical_exact_tiles import verify_against_reference_strip
        from darsia.fluidflower.phase_validation import predict_image_tiled
        from darsia.fluidflower.tools.phase_inference_benchmark_backends import partition
        from threadpoolctl import threadpool_limits
        import cv2
        started = time.perf_counter()
        key = self.cache.key(current, previous) if self.cache is not None else None
        # An explicit audit must exercise the current implementation; a cached
        # production output is not a receipt of a successful strip comparison.
        cached = self.cache.load(key) if self.cache is not None and not self.verify_numerics else None
        if cached is not None:
            heads, _ = cached
            self.report['frames'].append(dict(seconds=time.perf_counter()-started,
                optical_cache_hit=True, optical_cache_key=key))
            return partition(np.stack([heads['co2_presence_probability'],
                heads['gas_probability_conditional_on_co2']], axis=-1))
        # Tile workers already supply parallelism; nested BLAS/OpenCV teams
        # would multiply the requested CPU budget.
        old_threads = cv2.getNumThreads()
        try:
            cv2.setNumThreads(1)
            with threadpool_limits(limits=1):
                if not self.fallback:
                    heads, evaluated, _ = self.tiles.predict(current, previous)
                    if self.verify_numerics and not self.checked:
                        checks = [verify_against_reference_strip(self.tiles, current, previous, heads, evaluated, row=row)
                                  for row in sorted({0, max(0, current.shape[0] - 64)})]
                        self.fallback = not all(c['exact'] and c['compared_pixels'] for c in checks)
                        self.report['tile_verification'] = checks
                        self.checked = True
                if self.fallback:
                    heads = predict_image_tiled(self.model, current, self.baseline,
                                               previous_rgb=previous, radiometry_profile=self.profile)
                    self.report['backend'] = 'cpu-reference-strips'
        finally:
            cv2.setNumThreads(old_threads)
        if self.cache is not None:
            self.cache.store(key, heads)
        self.report['frames'].append(dict(seconds=time.perf_counter() - started,
            optical_cache_hit=False, optical_cache_key=key))
        return partition(np.stack([heads['co2_presence_probability'],
                                   heads['gas_probability_conditional_on_co2']], axis=-1))

    def close(self):
        self._tiles = None


def make_predictor(model, baseline, report, *, backend='auto', threads=1,
                   prefetch=False, parity_threads=1, optical_cache_root=None,
                   optical_cache_max_bytes=None, verify_numerics=False):
    if backend not in ('auto', 'cpu', 'cuda'):
        raise ValueError('Backend must be auto, cpu, or cuda')
    reason = None
    if backend != 'cpu':
        # Only preflight failures allow automatic CPU selection. Never hide a
        # numerical parity failure or a failing prediction behind a fallback.
        try:
            import cupy as cp
            from darsia.fluidflower.tools.phase_inference_benchmark_backends import CudaFeaturePredictor
            CudaFeaturePredictor.checked_contract(model)
            if cp.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError('No CUDA device')
            free, _ = cp.cuda.runtime.memGetInfo()
            needed = baseline.shape[0] * baseline.shape[1] * 260 + 512 * 1024 ** 2
            if free < needed:
                raise RuntimeError(f'CUDA memory estimate {needed} exceeds available {free}')
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            if backend == 'cuda':
                raise RuntimeError(f'Requested CUDA backend unavailable: {error}') from error
            reason = f'{type(error).__name__}: {error}'
        else:
            from darsia.fluidflower.tools.materialize_phase_iteration import CheckedCuda
            report['backend'] = 'cuda-float32-checked' if verify_numerics else 'cuda-float32'
            report['verify_numerics'] = bool(verify_numerics)
            report.setdefault('runs', [])
            if prefetch:
                from darsia.fluidflower.tools.phase_optical_prefetch import QueuedCuda
                return QueuedCuda(model, baseline, report, parity_threads=parity_threads,
                    optical_cache_root=optical_cache_root, optical_cache_max_bytes=optical_cache_max_bytes,
                    verify_numerics=verify_numerics)
            return CheckedCuda(model, baseline, report, optical_cache_root=optical_cache_root,
                               optical_cache_max_bytes=optical_cache_max_bytes,
                               verify_numerics=verify_numerics)
    report['backend'] = 'cpu-tiles-checked' if verify_numerics else 'cpu-tiles'
    report['verify_numerics'] = bool(verify_numerics)
    report['cpu_selection_reason'] = reason or 'CPU requested'
    return CpuPredictor(model, baseline, report, threads=threads, optical_cache_root=optical_cache_root,
                        optical_cache_max_bytes=optical_cache_max_bytes, verify_numerics=verify_numerics)
