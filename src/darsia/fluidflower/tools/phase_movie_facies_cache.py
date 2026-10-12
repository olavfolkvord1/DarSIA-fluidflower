"""Reuse exact phase intermediates in isolated movie worker processes.

Fixed sand geometry and Gaussian denominators persist across images. The
coarse-to-native support result is shared only between boundary stages of one
observation; a new causal support array invalidates it. Optical inputs,
thresholds and phase history remain with the unchanged scientific runtime.
"""
from __future__ import annotations

from collections import OrderedDict

import cv2
import numpy as np


class FrozenFaciesAverage:
    """Exact equivalent of smooth_within_facies for one immutable geometry."""

    def __init__(self, domain, facies, *, max_scales=2):
        if domain.dtype != bool or domain.ndim != 2 or facies.shape != domain.shape:
            raise ValueError("Expected a boolean domain and matching fixed facies")
        if max_scales < 1:
            raise ValueError("At least one cached scale is required")
        self.domain, self.facies = domain, facies
        self.max_scales = max_scales
        self.masks = None
        self.denominators = OrderedDict()
        self.hits = self.misses = 0

    def __call__(self, values, domain, facies, sigma_x, sigma_y):
        if domain is not self.domain or facies is not self.facies:
            raise ValueError("Cached normalization belongs to another geometry")
        if values.shape != domain.shape:
            raise ValueError("Values must match the frozen geometry")
        if self.masks is None:
            self.masks = [domain & (facies == label) for label in np.unique(facies[domain])]
        key = (float(sigma_x), float(sigma_y))
        denominators = self.denominators.get(key)
        if denominators is None:
            self.misses += 1
            denominators = []
            for mask in self.masks:
                weights = cv2.GaussianBlur(mask.astype(np.float32), (0, 0),
                    sigmaX=sigma_x, sigmaY=sigma_y)
                denominators.append(np.maximum(weights[mask], 1e-7))
            self.denominators[key] = denominators
            if len(self.denominators) > self.max_scales:
                self.denominators.popitem(last=False)
        else:
            self.hits += 1
            self.denominators.move_to_end(key)
        result = np.zeros(values.shape, np.float32)
        for mask, denominator in zip(self.masks, denominators):
            weighted = cv2.GaussianBlur(np.where(mask, values, 0).astype(np.float32),
                (0, 0), sigmaX=sigma_x, sigmaY=sigma_y)
            result[mask] = weighted[mask] / denominator
        return result

    @property
    def retained_bytes(self):
        return sum(a.nbytes for a in self.masks or []) + sum(
            a.nbytes for group in self.denominators.values() for a in group)


class FaciesCacheInstallation:
    """Process-local dispatch, falling back for any other geometry.

    The caller owns the registered immutable context. Forked phase histories
    retain that same context, so they can safely share these fixed arrays.
    """

    def __init__(self, context):
        from darsia.fluidflower import phase_presence_context
        self.module = phase_presence_context
        self.original = self.module.smooth_within_facies
        self.cache = FrozenFaciesAverage(context.domain, context.facies)

        def dispatch(values, domain, facies, sigma_x, sigma_y):
            if domain is self.cache.domain and facies is self.cache.facies:
                return self.cache(values, domain, facies, sigma_x, sigma_y)
            return self.original(values, domain, facies, sigma_x, sigma_y)

        self.dispatch = dispatch
        self.module.smooth_within_facies = dispatch

    def close(self):
        if self.module.smooth_within_facies is self.dispatch:
            self.module.smooth_within_facies = self.original


def install_facies_cache(context):
    return FaciesCacheInstallation(context)


class FrozenMaterialAverage:
    """Bit-exact connected-material averaging with immutable geometry cached.

    Component labels, bounded local masks and Gaussian denominators depend only
    on the registered sand geometry and requested physical scale. Numerators
    still run on every image. In particular, singletons use the OpenCV blur:
    copying their input value directly changes some float32 outputs by one ULP.
    """

    def __init__(self, domain, facies, *, max_scales=3):
        if domain.dtype != bool or domain.ndim != 2 or facies.shape != domain.shape:
            raise ValueError("Expected a boolean domain and matching fixed facies")
        self.domain, self.facies = domain, facies
        self.max_scales = max_scales
        self.components = None
        self.denominators = OrderedDict()
        self.hits = self.misses = 0

    def _prepare_components(self):
        if self.components is not None:
            return
        components = []
        for material in np.unique(self.facies[self.domain]):
            mask = self.domain & (self.facies == material)
            count, labels, stats, _ = cv2.connectedComponentsWithStats(
                mask.astype(np.uint8), connectivity=4)
            for component in range(1, count):
                x, y, width, height, _ = stats[component]
                region = np.s_[y:y+height, x:x+width]
                components.append((region, labels[region] == component))
        self.components = components

    def __call__(self, values, domain, facies, sigma_x, sigma_y):
        if domain is not self.domain or facies is not self.facies:
            raise ValueError("Cached material geometry belongs to another sequence")
        if values.shape != domain.shape:
            raise ValueError("Values must match the fixed material geometry")
        self._prepare_components()
        key = (float(sigma_x), float(sigma_y))
        denominators = self.denominators.get(key)
        if denominators is None:
            self.misses += 1
            denominators = [np.maximum(cv2.GaussianBlur(local.astype(np.float32), (0, 0),
                sigmaX=sigma_x, sigmaY=sigma_y, borderType=cv2.BORDER_CONSTANT)[local], 1e-7)
                for _, local in self.components]
            self.denominators[key] = denominators
            if len(self.denominators) > self.max_scales:
                self.denominators.popitem(last=False)
        else:
            self.hits += 1
            self.denominators.move_to_end(key)
        result = np.zeros(values.shape, np.float32)
        for (region, local), denominator in zip(self.components, denominators):
            numerator = cv2.GaussianBlur(np.where(local, values[region], 0).astype(np.float32),
                (0, 0), sigmaX=sigma_x, sigmaY=sigma_y, borderType=cv2.BORDER_CONSTANT)
            result[region][local] = numerator[local] / denominator
        return result

    @property
    def retained_bytes(self):
        self._prepare_components()
        return sum(local.nbytes for _, local in self.components) + sum(
            part.nbytes for group in self.denominators.values() for part in group)


class MaterialCacheInstallation:
    def __init__(self, context):
        from darsia.fluidflower import phase_aqueous_boundary
        self.module = phase_aqueous_boundary
        self.original = self.module._material_average
        self.cache = FrozenMaterialAverage(context.domain, context.facies)

        def dispatch(values, domain, facies, sigma_x, sigma_y):
            if domain is self.cache.domain and facies is self.cache.facies:
                return self.cache(values, domain, facies, sigma_x, sigma_y)
            return self.original(values, domain, facies, sigma_x, sigma_y)

        self.dispatch = dispatch
        self.module._material_average = dispatch

    def close(self):
        if self.module._material_average is self.dispatch:
            self.module._material_average = self.original


def install_material_cache(context):
    return MaterialCacheInstallation(context)


class FrozenContinuousSupport:
    """Share one unchanged coarse-to-native support interpolation per frame.

    The v1 and v3 boundary stages request this same field consecutively. A new
    `previous_supported` array marks the next observation, including a replayed
    reference, so a later frame cannot receive an earlier frame's support.
    """

    def __init__(self, context, original):
        self.context = context
        self.original = original
        self.parent = self.source = self.result = None
        self.hits = self.misses = 0

    def __call__(self, sequence):
        from darsia.fluidflower.phase_aqueous_boundary_runtime import _presence_sequence

        parent = _presence_sequence(sequence)
        if parent.context is not self.context:
            return self.original(sequence)
        source = parent.previous_supported
        if parent is self.parent and source is self.source:
            self.hits += 1
            result = self.result
            self.parent = self.source = self.result = None
            return result
        self.misses += 1
        result = self.original(sequence)
        self.parent, self.source, self.result = parent, source, result
        return result

    @property
    def retained_bytes(self):
        return 0 if self.result is None else self.result.nbytes


class SupportCacheInstallation:
    def __init__(self, context):
        from darsia.fluidflower import phase_aqueous_boundary_runtime

        self.module = phase_aqueous_boundary_runtime
        self.original = self.module.continuous_supported_mask
        self.cache = FrozenContinuousSupport(context, self.original)
        self.module.continuous_supported_mask = self.cache

    def close(self):
        if self.module.continuous_supported_mask is self.cache:
            self.module.continuous_supported_mask = self.original


def install_support_cache(context):
    return SupportCacheInstallation(context)
