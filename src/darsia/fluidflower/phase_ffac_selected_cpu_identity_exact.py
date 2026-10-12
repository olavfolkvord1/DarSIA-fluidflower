"""Exact saved spacing and native-only temporal alignment conventions.

This module wraps the frozen corrected operator without editing it. Decoding
and feature-support masks remain intact. Only the historical native previous
alignment omits its explicit mask; extended references keep theirs.
"""
from __future__ import annotations

from contextlib import nullcontext
from unittest.mock import patch

import numpy as np

from .phase_ffac_selected_cpu_identity import IdentityEvaluator, CapturedCentre
from .phase_ffac_selected_inference_corrected import SelectedPhaseEvaluator
from .phase_ffac_selected_phase import bound_json


class ExactConventionCentre:
    def __init__(self, inner, *, mode, frozen_spacing_receipt=None):
        if mode not in ('spacing_only', 'both'):
            raise ValueError('Unknown exact-convention identity mode')
        self.inner, self.mode = inner, mode
        self.frozen_spacing_receipt = frozen_spacing_receipt

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def infer(self, *args, **kwargs):
        from .tools import ffac_v5_temporal_alignment as temporal
        receipt = kwargs.get('probability_receipt')
        spacing_binding = None
        if receipt is None and self.frozen_spacing_receipt is not None:
            spacing_binding = self.frozen_spacing_receipt
            receipt = bound_json(spacing_binding)
        if receipt is None:
            raise ValueError('Identity spacing requires the bound original probability receipt')
        spacing = receipt['metadata']['spacing_m']
        if (len(spacing) != 2 or not np.isfinite(spacing).all()
                or np.any(np.asarray(spacing) <= 0)):
            raise ValueError('Saved original y/x spacing is invalid')
        exact_kwargs = dict(kwargs, spacing_m=tuple(spacing))
        original_alignment = temporal.align_temporal_inputs

        def exact_alignment(*positional, **named):
            native_previous = (named.get('previous') is not None
                and tuple(named.get('previous_origin_xy', (0, 0))) == (0, 0)
                and np.asarray(named['previous']).shape == np.asarray(named['current']).shape)
            if native_previous:
                named = dict(named, previous_source_valid_mask=None)
            return original_alignment(*positional, **named)

        context = (patch.object(temporal, 'align_temporal_inputs', exact_alignment)
                   if self.mode == 'both' else nullcontext())
        with context:
            result = self.inner.infer(*args, **exact_kwargs)
        result.provenance['exact_conventions'] = {
            'mode': self.mode, 'spacing_m': list(spacing),
            'spacing_source': 'matched original stored probability receipt metadata.spacing_m',
            'fallback_spacing_receipt': spacing_binding,
            'previous_native_alignment_mask': 'absent' if self.mode == 'both' else 'unchanged',
            'extended_previous_alignment_mask': 'unchanged',
            'decoder_validity_mask': 'unchanged', 'feature_support_mask': 'unchanged',
            'frozen_model_or_labels_changed': False,
        }
        return result


class ExactIdentityEvaluator(IdentityEvaluator):
    def __init__(self, root, plan, run, *, mode):
        super().__init__(root, plan, run)
        self.centre = CapturedCentre(ExactConventionCentre(self.centre.inner, mode=mode))


class ExactSelectedPhaseEvaluator(SelectedPhaseEvaluator):
    """Same frozen full chain, with BOTH conventions for approved Policy A."""
    def __init__(self, root, plan, run, *, backend='cpu'):
        if backend != 'cpu':
            raise ValueError('Exact-convention phase inference is CPU only')
        super().__init__(root, plan, run, backend=backend)
        baseline_receipt = self.baseline_source['parent_source']['retained_source_cache']['parent_probability_receipt']
        bound_json(baseline_receipt)
        self.centre = ExactConventionCentre(self.centre, mode='both',
            frozen_spacing_receipt=baseline_receipt)

