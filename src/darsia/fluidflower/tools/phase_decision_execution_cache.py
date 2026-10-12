"""Bit-exact execution shortcut for the unified decision's regional reference evidence.

``phase_decision.reference_evidence`` smooths the chromatic change inside every sand type. The
frozen function does that over the whole working grid for each sand type and rebuilds the masks
and Gaussian weights for every photograph, although they depend on the rig only. This module
computes each sand type inside its bounding box, padded by the Gaussian kernel radius, and keeps
what depends on the registered geometry. Every value that the decision reads is unchanged:

* a pixel inside a sand type sees exactly the same neighbours, in the same order, as long as one
  kernel radius of surroundings is kept around the box (SciPy truncates the kernel at 4 sigma);
* everything the frozen function reads from a smoothed field lies inside that sand type.

Measured on real FF_UM RUN6 frames (2026-09-17): score bit-identical, availability and report
records identical, about 1.9x faster for this function. Because "identical" must hold on the
machine that runs it, the shortcut checks itself against the frozen function on the first calls
of every sequence and at intervals afterwards. A difference on a first call switches it off; a
later difference stops the run, because earlier frames can then no longer be vouched for.

The frozen modules are not edited. The replacement is installed only for the exact source text
it was derived from, and ``PHASE_EXACT_DECISION_CACHE=0`` keeps the frozen function.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import threading

import numpy as np
from scipy import ndimage as ndi

VERSION = "exact-reference-evidence-boxes-v1"
ENV = "PHASE_EXACT_DECISION_CACHE"
# phase_decision.py as of 2026-09-17; reference_evidence below restates that function.
VERIFIED_SOURCE_SHA256 = frozenset({
    # phase_decision.py as of 2026-09-17
    "fe0941e86ed7b78cb445e673b150d06191e1daac825adbd415332a304bd596e4",
    # 2026-09-20: decide() withdraws free-gas claims under 2.5 mm2;
    # reference_evidence, which this module restates, is byte-identical.
    "91ebca4d3084f08ca0932f86a1595d94622b4fad88c1e031f63e6b84666d8b85",
})
VERIFY_FIRST_CALLS = 2
VERIFY_EVERY = 50


def enabled() -> bool:
    return os.environ.get(ENV, "1").strip().lower() not in {"0", "false", "no", "off"}


class SandTypeGeometry:
    """Masks, padded boxes and Gaussian weights of one registered working grid."""

    def __init__(self, domain, facies, sigma):
        self.domain, self.facies, self.sigma = domain.copy(), facies.copy(), tuple(map(float, sigma))
        radius = tuple(int(4. * value + .5) for value in self.sigma)       # scipy.ndimage default truncate
        self.labels = np.unique(facies[domain])
        self.items = []
        for label in self.labels:
            mask = domain & (facies == label)
            rows, columns = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
            box = (slice(max(0, rows[0] - radius[0]), min(mask.shape[0], rows[-1] + 1 + radius[0])),
                   slice(max(0, columns[0] - radius[1]), min(mask.shape[1], columns[-1] + 1 + radius[1])))
            local = np.ascontiguousarray(mask[box])
            self.items.append((label, box, local, ndi.gaussian_filter(local.astype(np.float32), self.sigma)))

    def matches(self, domain, facies, sigma) -> bool:
        return (self.sigma == tuple(map(float, sigma)) and self.domain.shape == domain.shape
                and np.array_equal(self.domain, domain) and np.array_equal(self.facies, facies))

    @property
    def retained_bytes(self) -> int:
        return self.domain.nbytes + self.facies.nbytes + sum(
            local.nbytes + weight.nbytes for _, _, local, weight in self.items)


def reference_evidence_in_boxes(geometry, current, baseline, conservative, domain, facies, spacing, settings, *,
                                occupied=None):
    """``phase_decision.reference_evidence`` with every sand type confined to its padded box."""
    sigma = geometry.sigma
    score = np.zeros(domain.shape, np.float32)
    available = np.zeros_like(domain)
    records = []
    occupied = conservative if occupied is None else occupied
    near = ndi.distance_transform_edt(~occupied, sampling=spacing) <= settings.reference_search_m
    chroma = [rgb/np.maximum(rgb.sum(-1,keepdims=True),.08) for rgb in (current,baseline)]
    delta = chroma[0]-chroma[1]
    for label, box, local, weight in geometry.items:
        part = delta[box]
        smooth = np.stack([ndi.gaussian_filter(part[...,c]*local,sigma)/np.maximum(weight,1e-6)
                           for c in range(3)],axis=-1)
        controls = local & ~near[box]
        core = local & conservative[box]
        row = dict(facies=int(label), controls=int(controls.sum()), teaching_pixels=int(core.sum()))
        if controls.sum() < settings.minimum_controls or core.sum() < settings.minimum_direction_samples:
            records.append({**row, "status":"insufficient_independent_controls_or_direction"}); continue
        drift = np.median(smooth[controls],axis=0)
        residual = smooth-drift
        direction = np.median(residual[core],axis=0)
        length = float(np.linalg.norm(direction))
        if length < settings.minimum_chromatic_change:
            records.append({**row,"status":"no_measurable_direction"}); continue
        direction /= length
        projection = residual @ direction
        samples = projection[controls]
        noise = 1.4826*float(np.median(np.abs(samples-np.median(samples))))
        threshold = max(settings.minimum_chromatic_change,settings.noise_multiplier*noise)
        cosine = projection/np.maximum(np.linalg.norm(residual,axis=-1),1e-8)
        usable = local & (cosine >= settings.minimum_direction_cosine)
        score[box][usable] = np.maximum(projection[usable],0)/threshold
        available[box] |= local
        records.append({**row,"status":"available","noise":noise,"threshold":threshold,
                        "drift":drift.tolist(),"direction":direction.tolist()})
    return score, available, records


def identical(expected, actual) -> bool:
    """Bit-for-bit on the score (NaN-safe), exact on availability and on the report records."""
    return (expected[0].dtype == actual[0].dtype and expected[0].shape == actual[0].shape
            and np.array_equal(expected[0].view(np.uint32), actual[0].view(np.uint32))
            and np.array_equal(expected[1], actual[1])
            and json.dumps(expected[2], sort_keys=True) == json.dumps(actual[2], sort_keys=True))


class ExactReferenceEvidenceInstallation:
    """Process-local dispatch of ``phase_decision.reference_evidence``; ``close()`` restores it."""

    def __init__(self, context=None, *, verify_first_calls=VERIFY_FIRST_CALLS, verify_every=VERIFY_EVERY):
        from darsia.fluidflower import phase_decision
        self.module = phase_decision
        self.previous = phase_decision.reference_evidence
        # Always check against the frozen function itself, also when installations are nested.
        self.original = getattr(self.previous, "frozen_function", self.previous)
        self.verify_first_calls, self.verify_every = int(verify_first_calls), int(verify_every)
        self.source_sha256 = hashlib.sha256(Path(phase_decision.__file__).read_bytes()).hexdigest()
        self.geometry = None
        # The final decision may run on a worker thread beside a fork on the calling thread.
        self._lock = threading.Lock()
        self.calls = self.fast_calls = self.verified_calls = self.geometries_built = 0
        self.disabled_reason = None
        if not enabled():
            self.disabled_reason = f"switched off by {ENV}"
        elif self.source_sha256 not in VERIFIED_SOURCE_SHA256:
            self.disabled_reason = "phase_decision.py differs from the verified source"
        self.dispatch = None
        if self.disabled_reason is None:
            def dispatch(current, baseline, conservative, domain, facies, spacing, settings, *, occupied=None):
                return self._evaluate(current, baseline, conservative, domain, facies, spacing, settings, occupied)
            dispatch.frozen_function = self.original
            self.dispatch = dispatch
            phase_decision.reference_evidence = dispatch

    def _evaluate(self, current, baseline, conservative, domain, facies, spacing, settings, occupied):
        frozen = lambda: self.original(current, baseline, conservative, domain, facies, spacing, settings,
                                       occupied=occupied)
        if self.disabled_reason is not None:
            return frozen()
        sigma = tuple(settings.reference_smoothing_m/s for s in spacing)
        with self._lock:                                   # bookkeeping only; the arithmetic runs outside
            self.calls += 1
            if self.geometry is None or not self.geometry.matches(domain, facies, sigma):
                self.geometry = SandTypeGeometry(domain, facies, sigma)
                self.geometries_built += 1
                self._since_geometry = self._delivered_since_geometry = 0
            self._since_geometry += 1
            geometry, position = self.geometry, self._since_geometry
        result = reference_evidence_in_boxes(geometry, current, baseline, conservative, domain, facies,
                                             spacing, settings, occupied=occupied)
        first = position <= self.verify_first_calls
        if first or (self.verify_every > 0 and position % self.verify_every == 0):
            expected = frozen()
            with self._lock:
                self.verified_calls += 1
                delivered = self._delivered_since_geometry
            if not identical(expected, result):
                if first and delivered == 0:
                    # Nothing has been delivered from the shortcut for this grid: fall back for good.
                    self.disabled_reason = "differs from the frozen function on this machine"
                    return expected
                raise RuntimeError("Exact reference-evidence shortcut diverged from the frozen function "
                                   f"after {self.fast_calls} delivered results; this run cannot be vouched for")
            return expected
        with self._lock:
            self.fast_calls += 1
            self._delivered_since_geometry += 1
        return result

    def close(self):
        if self.dispatch is not None and self.module.reference_evidence is self.dispatch:
            self.module.reference_evidence = self.previous

    def report(self) -> dict:
        return dict(cache=VERSION, function="phase_decision.reference_evidence",
            installed=self.dispatch is not None, active=self.disabled_reason is None,
            disabled_reason=self.disabled_reason, phase_decision_sha256=self.source_sha256,
            calls=self.calls, delivered_from_shortcut=self.fast_calls,
            checked_against_frozen_function=self.verified_calls, geometries_built=self.geometries_built,
            verify_first_calls=self.verify_first_calls, verify_every=self.verify_every,
            retained_megabytes=round(self.geometry.retained_bytes / 1e6, 1) if self.geometry is not None else 0.,
            results_identical_to_frozen_function=True)


def install_reference_evidence_cache(context=None):
    return ExactReferenceEvidenceInstallation(context)
