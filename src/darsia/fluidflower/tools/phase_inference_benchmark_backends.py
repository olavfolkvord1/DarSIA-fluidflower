"""Experimental inference backends. Not imported by Studio or production code."""
from __future__ import annotations

import cv2
import numpy as np
from scipy.special import expit

from darsia.fluidflower.phase_features import (
    BASELINE_RELATIVE_FEATURE_VERSION,
    GENERIC_FEATURE_VERSION,
    RELATIVE_BRIGHT_GRAIN_FEATURE_VERSION,
    OpticalFeatureContract,
    relative_bright_grain_columns,
    SPATIAL_CONTEXT_FRACTION,
)
from darsia.fluidflower.phase_learning import HierarchicalPhaseModel


def packed_parameters(model):
    heads = (model.co2_presence, model.gas_within_co2)
    weights = np.column_stack([h.coefficients / h.scale for h in heads])
    bias = np.array([h.intercept - h.mean @ weights[:, i] for i, h in enumerate(heads)])
    return weights, bias


def partition(values):
    co2, gas = values[..., 0], values[..., 1]
    free = co2 * gas
    return {
        'co2_presence_probability': co2,
        'gas_probability_conditional_on_co2': gas,
        'untouched_water_probability': 1 - co2,
        'aqueous_co2_probability': co2 * (1 - gas),
        'free_co2_gas_probability': free,
    }


class PackedPredictor:
    """Keep original CPU features, fold scaling into two packed linear heads."""
    predict_image = HierarchicalPhaseModel.predict_image

    def __init__(self, model, backend):
        self.model = model
        self.backend = backend
        weights, bias = packed_parameters(model)
        if backend == 'cuda-heads':
            import cupy as cp
            self.xp = cp
            self.weights = cp.asarray(weights, dtype=cp.float32)
            self.bias = cp.asarray(bias, dtype=cp.float32)
        else:
            self.xp = np
            dtype = np.float64 if backend == 'cpu-packed64' else np.float32
            self.weights, self.bias = weights.astype(dtype), bias.astype(dtype)

    def __getattr__(self, key):
        return getattr(self.model, key)

    def predict_features(self, features):
        if self.backend == 'cuda-heads':
            cp = self.xp
            z = cp.asarray(features, dtype=cp.float32) @ self.weights + self.bias
            values = cp.asnumpy(1 / (1 + cp.exp(-z)))
        else:
            values = expit(features @ self.weights + self.bias).astype(np.float32)
        return partition(values)


class CudaFeaturePredictor:
    """Benchmark only: full CUDA float32 features, cached baseline, CPU physics.

    Supports generic v3/v4/v5 contracts, with or without dynamic features.
    Chemistry/reference adapters must never fall back silently.
    """
    @staticmethod
    def checked_contract(model):
        contract = model.feature_contract
        supported = tuple(
            OpticalFeatureContract(generic_version=version, include_dynamic=dynamic)
            for version in (BASELINE_RELATIVE_FEATURE_VERSION, GENERIC_FEATURE_VERSION, RELATIVE_BRIGHT_GRAIN_FEATURE_VERSION)
            for dynamic in (False, True)
        )
        if contract not in supported or any(
            head.radiometry_profile is not None
            for head in (model.co2_presence, model.gas_within_co2)
        ):
            raise ValueError('Benchmark requires generic v3/v4/v5, no adapter, already corrected RGB')
        return contract

    def __init__(self, model, baseline, *, block_rows=384, texture_sigma=2.0,
                 input_encoding='review-uint8'):
        self.feature_contract = self.checked_contract(model)
        if input_encoding not in ('review-uint8', 'v5-unit-float32'):
            raise ValueError('Unknown optical input encoding')
        self.input_encoding = input_encoding
        import cupy as cp
        from cupyx.scipy.ndimage import correlate1d
        self.cp, self.correlate1d = cp, correlate1d
        self.block_rows, self.texture_sigma = block_rows, texture_sigma
        self.spatial_sigma = max(texture_sigma, SPATIAL_CONTEXT_FRACTION * min(baseline.shape[:2]))
        weights, bias = packed_parameters(model)
        self.weights, self.bias = cp.asarray(weights, dtype=cp.float32), cp.asarray(bias, dtype=cp.float32)
        self.kernels = {}
        self.baseline = self.prepare_rgb(baseline)
        self.previous = None
        self.previous_key = None
        cp.cuda.get_current_stream().synchronize()

    def blur(self, values, sigma):
        cp = self.cp
        if sigma not in self.kernels:
            # OpenCV's automatically sized float-image Gaussian: radius ~4 sigma.
            size = int(round(float(sigma) * 8 + 1)) | 1
            self.kernels[sigma] = cp.asarray(cv2.getGaussianKernel(size, sigma, cv2.CV_32F).ravel())
        kernel = self.kernels[sigma]
        row = self.correlate1d(values, kernel, axis=1, mode='mirror')
        return self.correlate1d(row, kernel, axis=0, mode='mirror')

    def prepare_rgb(self, image):
        cp = self.cp
        source = np.asarray(image)
        if source.ndim != 3 or source.shape[2] != 3:
            raise ValueError('Optical input must be HWC RGB')
        if self.input_encoding == 'review-uint8':
            if source.dtype != np.uint8:
                raise ValueError('Review optical input requires uint8 corrected pixels')
            rgb = cp.asarray(source, dtype=cp.float32) / cp.float32(255)
        else:
            if (source.dtype != np.float32 or not np.isfinite(source).all()
                    or np.any(source < 0) or np.any(source > 1)):
                raise ValueError('V5 optical input requires finite float32 unit RGB')
            rgb = cp.asarray(source, dtype=cp.float32)
        gray = cp.mean(rgb, axis=2, dtype=cp.float32)
        smooth = self.blur(gray, self.texture_sigma)
        contrast = gray - smooth
        texture = cp.abs(contrast)
        fraction = rgb / (cp.sum(rgb, axis=2, keepdims=True) + cp.float32(1e-4))
        safe = cp.maximum(rgb, cp.float32(1e-4))
        ratios = cp.stack([cp.log(safe[..., 0] / safe[..., 1]),
                           cp.log(safe[..., 1] / safe[..., 2]),
                           cp.log(safe[..., 0] / safe[..., 2])], axis=-1)
        return {'rgb': rgb, 'gray': gray, 'smooth': smooth, 'texture': texture, 'signed_contrast': contrast,
                'fraction': fraction, 'ratios': ratios}

    def set_baseline(self, baseline):
        """Rebind a frame-aligned V5 baseline without rebuilding model weights.

        The caller must certify its geometry and pass the corrected baseline
        warped into the current native frame.  Cached previous pixels are
        cleared because their coordinate system is no longer valid.
        """
        if self.input_encoding != 'v5-unit-float32':
            raise ValueError('Per-frame baseline rebinding is V5-only')
        if tuple(np.asarray(baseline).shape[:2]) != tuple(self.baseline['gray'].shape):
            raise ValueError('Aligned baseline shape changed')
        self.baseline = self.prepare_rgb(baseline)
        self.previous = None
        self.previous_key = None

    def feature_blocks(self, current, previous, *, dynamic_available=True):
        cp, base = self.cp, self.baseline
        delta = current['rgb'] - base['rgb']
        dynamic = current['rgb'] - previous['rgb']
        norm = cp.sqrt(cp.sum(delta * delta, axis=2))
        dnorm = cp.sqrt(cp.sum(dynamic * dynamic, axis=2))
        mean = self.blur(norm, self.spatial_sigma)
        std = cp.sqrt(cp.maximum(self.blur(norm * norm, self.spatial_sigma) - mean * mean, cp.float32(0)))
        dmean = self.blur(dnorm, self.spatial_sigma)
        height, width = norm.shape
        for start in range(0, height, self.block_rows):
            stop = min(start + self.block_rows, height)
            sl = slice(start, stop)
            texture = current['texture'][sl] - base['texture'][sl]
            blocks = [
                delta[sl], (current['gray'][sl] - base['gray'][sl])[..., None],
                current['fraction'][sl] - base['fraction'][sl],
                current['ratios'][sl] - base['ratios'][sl],
                texture[..., None], cp.abs(texture)[..., None], norm[sl, :, None],
                mean[sl, :, None], std[sl, :, None],
                (mean[sl] / (std[sl] + cp.float32(1e-4)))[..., None],
            ]
            if self.feature_contract.generic_version in {GENERIC_FEATURE_VERSION, RELATIVE_BRIGHT_GRAIN_FEATURE_VERSION}:
                baseline_contrast = base['signed_contrast'][sl]
                current_contrast = current['signed_contrast'][sl]
                baseline_bright = cp.maximum(baseline_contrast, cp.float32(0))
                current_bright = cp.maximum(current_contrast, cp.float32(0))
                blocks.extend([
                    base['gray'][sl, :, None], baseline_contrast[..., None],
                    current_contrast[..., None],
                    (current_contrast - baseline_contrast)[..., None],
                    cp.minimum(baseline_bright, current_bright)[..., None],
                    cp.maximum(current_bright - baseline_bright, cp.float32(0))[..., None],
                ])
            if self.feature_contract.generic_version == RELATIVE_BRIGHT_GRAIN_FEATURE_VERSION:
                blocks.extend(value[..., None] for value in relative_bright_grain_columns(
                    base['gray'][sl], base['smooth'][sl],
                    current['gray'][sl], current['smooth'][sl], xp=cp))
            if self.feature_contract.include_dynamic:
                blocks.append(cp.full((stop - start, width, 1), int(dynamic_available), dtype=cp.float32))
                if dynamic_available:
                    blocks.extend([dynamic[sl], (current['gray'][sl] - previous['gray'][sl])[..., None],
                        (current['texture'][sl] - previous['texture'][sl])[..., None],
                        dnorm[sl, :, None], dmean[sl, :, None]])
                else:
                    blocks.append(cp.zeros((stop - start, width, 7), dtype=cp.float32))
            yield start, stop, cp.concatenate(blocks, axis=2).reshape(-1, len(self.feature_contract.feature_names))

    def predict(self, current_rgb, previous_rgb, *, current_key=None, previous_key=None):
        cp = self.cp
        current = self.prepare_rgb(current_rgb)
        if previous_rgb is None:
            previous = self.baseline
        elif previous_key is None or self.previous_key != previous_key:
            previous = self.prepare_rgb(previous_rgb)
        else:
            previous = self.previous
        height, width = current_rgb.shape[:2]
        output = np.empty((height, width, 2), dtype=np.float32)
        for start, stop, features in self.feature_blocks(current, previous, dynamic_available=previous_rgb is not None):
            z = features @ self.weights + self.bias
            output[start:stop] = cp.asnumpy(1 / (1 + cp.exp(-z))).reshape(stop-start, width, 2)
        self.previous, self.previous_key = current, current_key
        cp.cuda.get_current_stream().synchronize()
        # The production phase layer needs these two fields, not five GPU transfers.
        return {'co2_presence_probability': output[..., 0],
                'gas_probability_conditional_on_co2': output[..., 1]}


class CpuFeaturePredictor(CudaFeaturePredictor):
    """CPU control with the same caching and lean feature layout as CUDA."""
    def __init__(self, model, baseline, *, block_rows=384, texture_sigma=2.0,
                 input_encoding='review-uint8', feature_block_rows=None):
        self.feature_contract = self.checked_contract(model)
        if input_encoding not in ('review-uint8', 'v5-unit-float32'):
            raise ValueError('Unknown optical input encoding')
        self.input_encoding = input_encoding
        self.cp = np
        self.block_rows, self.texture_sigma = block_rows, texture_sigma
        # Opt-in CPU batch-size cap only. The full-image preparation and all
        # per-pixel arithmetic stay in the original feature implementation.
        # None preserves the historical caller's block size exactly.
        if feature_block_rows is not None:
            if (not isinstance(feature_block_rows, int)
                    or isinstance(feature_block_rows, bool)
                    or feature_block_rows < 1):
                raise ValueError('feature_block_rows must be a positive integer or None')
            self.block_rows = min(self.block_rows, feature_block_rows)
        self.feature_block_rows = feature_block_rows
        self.spatial_sigma = max(texture_sigma, SPATIAL_CONTEXT_FRACTION * min(baseline.shape[:2]))
        weights, bias = packed_parameters(model)
        self.weights, self.bias = weights.astype(np.float32), bias.astype(np.float32)
        self.baseline = self.prepare_rgb(baseline)
        self.previous = None
        self.previous_key = None

    def blur(self, values, sigma):
        return cv2.GaussianBlur(values, (0, 0), sigmaX=float(np.float32(sigma)),
                                sigmaY=float(np.float32(sigma)))

    def predict(self, current_rgb, previous_rgb, *, current_key=None, previous_key=None):
        current = self.prepare_rgb(current_rgb)
        previous = (self.baseline if previous_rgb is None else
                    self.previous if previous_key is not None and self.previous_key == previous_key
                    else self.prepare_rgb(previous_rgb))
        height, width = current_rgb.shape[:2]
        output = np.empty((height, width, 2), dtype=np.float32)
        for start, stop, features in self.feature_blocks(current, previous, dynamic_available=previous_rgb is not None):
            output[start:stop] = expit(features @ self.weights + self.bias).reshape(stop-start, width, 2)
        self.previous, self.previous_key = current, current_key
        return {'co2_presence_probability': output[..., 0],
                'gas_probability_conditional_on_co2': output[..., 1]}
