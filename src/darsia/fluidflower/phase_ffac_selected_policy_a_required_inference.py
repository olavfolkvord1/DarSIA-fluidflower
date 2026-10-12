"""Policy A inference with the independently verified historical conventions.

The numerical stage wrapper is copied from the accepted required-identity
helper. Its only selected-image extension reads the original baseline's axes
metadata when no matching original probability receipt exists. It does not
substitute that baseline's probability storage/encoding into inference.
"""
from __future__ import annotations
from contextlib import ExitStack
from unittest.mock import patch
import numpy as np
from .phase_ffac_selected_cpu_identity import array_identity
from .phase_ffac_selected_inference_corrected import SelectedPhaseEvaluator
from .phase_ffac_selected_phase import bound_json
from .phase_ffac_rig_camera_runtime import bound_path
from .phase_ffac_selected_required_identity import require_frozen_gas_audit

STAGE_CONVENTION_SOURCE_SHA256 = "ebceacf58dee545231fb7295764867b4954c5b229c2d8ec22310250d13589a7a"

class RequiredConventionCentre:
    """Use optical inherited axes and G3 export axes from separate bindings."""
    def __init__(self, inner, inherited_conventions, frozen_spacing_receipt):
        self.inner = inner
        self.inherited_conventions = inherited_conventions
        self.frozen_spacing_receipt = frozen_spacing_receipt

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def infer(self, *args, **kwargs):
        from . import phase_ffac_selected_native_centre as native
        from .tools import ffac_v5_temporal_alignment as temporal
        probability = kwargs.get('probability_receipt')
        spacing_binding = None
        if probability is None:
            # Axes metadata only: never inject baseline's encoding receipt into
            # the genuinely new acquisition's numerical inference kwargs.
            spacing_binding = self.frozen_spacing_receipt
            probability = bound_json(spacing_binding)
        g3_spacing = probability['metadata']['spacing_m']
        inherited_spacing = self.inherited_conventions['spacing_m']
        for spacing in (g3_spacing, inherited_spacing):
            if len(spacing) != 2 or not np.isfinite(spacing).all() or min(spacing) <= 0:
                raise ValueError('A separately bound stage has invalid spacing')
        original_co2 = native.original_co2_labels
        original_alignment = temporal.align_temporal_inputs
        stage_state = {}

        def inherited_stage(*positional, **named):
            if len(positional) < 6:
                raise ValueError('Inherited stage positional contract changed')
            values = list(positional)
            if 'inherited_baseline' not in stage_state:
                raise ValueError('Inherited reference was not independently aligned')
            values[1] = stage_state['inherited_baseline']
            values[5] = tuple(inherited_spacing)
            return original_co2(*values, **named)

        def alignment(*positional, **named):
            native_previous = (named.get('previous') is not None
                and tuple(named.get('previous_origin_xy', (0, 0))) == (0, 0)
                and np.asarray(named['previous']).shape == np.asarray(named['current']).shape)
            if native_previous:
                named = dict(named, previous_source_valid_mask=None)
            inherited_named = dict(named)
            native_baseline = (tuple(named.get('baseline_origin_xy', (0, 0))) == (0, 0)
                and np.asarray(named['baseline']).shape == np.asarray(named['current']).shape)
            if native_baseline:
                inherited_named['baseline_source_valid_mask'] = None
            inherited_base, inherited_previous, inherited_receipt = original_alignment(*positional, **inherited_named)
            g3_base, g3_previous, g3_receipt = original_alignment(*positional, **named)
            # The historical warp does not normalize RGB by validity masks.
            # Check exact equality before sharing the optical two-head scores;
            # an unproven reference equivalence must never be assumed.
            same_baseline = array_identity(inherited_base, g3_base)
            same_previous = (inherited_previous is None and g3_previous is None
                or inherited_previous is not None and g3_previous is not None
                and array_identity(inherited_previous, g3_previous)['bit_identical'])
            if not same_baseline['bit_identical'] or not same_previous:
                raise ValueError('Independently aligned inherited and G3 RGB differ; sharing scores is forbidden')
            stage_state.update(inherited_baseline=inherited_base,
                inherited_alignment=inherited_receipt, g3_alignment=g3_receipt,
                baseline_rgb_bit_identity=same_baseline, previous_rgb_bit_identical=bool(same_previous))
            return g3_base, g3_previous, g3_receipt

        with ExitStack() as stack:
            stack.enter_context(patch.object(native, 'original_co2_labels', inherited_stage))
            stack.enter_context(patch.object(temporal, 'align_temporal_inputs', alignment))
            result = self.inner.infer(*args, **dict(kwargs, spacing_m=tuple(g3_spacing)))
        result.provenance['historical_stage_conventions'] = {
            'inherited_map': self.inherited_conventions,
            'g3_probability': {'spacing_m': list(g3_spacing),
                'fallback_axes_receipt': spacing_binding,
                'fallback_probability_injected_into_inference': False,
                'source': 'matched original probability receipt metadata.spacing_m'},
            'native_previous_alignment': 'historical absent explicit validity mask',
            'inherited_native_baseline_alignment': 'historical absent explicit validity mask',
            'g3_native_baseline_alignment': 'historical explicit source-valid mask',
            'extended_previous_alignment': 'retain bound mask',
            'decoder_and_feature_support_masks': 'unchanged',
            'independent_alignment_audit': {key: value for key, value in stage_state.items()
                if key != 'inherited_baseline'},
        }
        return result


class RequiredSelectedPhaseEvaluator(SelectedPhaseEvaluator):
    """Unchanged frozen P1 chain, admitted separately by required identity."""
    def __init__(self, root, plan, run, *, backend='cpu'):
        if backend != 'cpu':
            raise ValueError('Policy A required-convention inference is CPU only')
        super().__init__(root, plan, run, backend=backend)
        binding = plan['phase_campaign_binding']
        convention = binding['inherited_stage_conventions'][run]
        for key in ('postflight_manifest', 'context'):
            bound_path(convention[key])
        audit = bound_json(binding['frozen_gas_audit'])
        require_frozen_gas_audit(audit)
        if (audit.get('required_helper_binding_pending') is not False
                or audit.get('required_identity_helper_sha256') != STAGE_CONVENTION_SOURCE_SHA256
                or binding['code']['darsia.fluidflower.phase_ffac_selected_required_identity']['sha256'] != STAGE_CONVENTION_SOURCE_SHA256):
            raise ValueError('Policy A wrapper lacks the accepted exact stage-consumer audit')
        baseline_receipt = self.baseline_source['parent_source']['retained_source_cache']['parent_probability_receipt']
        bound_json(baseline_receipt)
        self.centre = RequiredConventionCentre(self.centre, convention, baseline_receipt)


__all__=['RequiredConventionCentre','RequiredSelectedPhaseEvaluator','STAGE_CONVENTION_SOURCE_SHA256']
