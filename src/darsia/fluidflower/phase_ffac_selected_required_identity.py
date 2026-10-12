"""Required CPU identity with independent historical stage conventions.

The inherited map is computed by the frozen numerical chain. Reading its
historical map is allowed only after inference, as expected comparison data.
"""
from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np

from .phase_ffac_selected_cpu_identity import CapturedCentre, array_identity
from .phase_ffac_selected_inference_corrected import SelectedPhaseEvaluator
from .phase_ffac_selected_phase import asset, bound_json
from .phase_ffac_rig_camera_runtime import bound_path

CONTRACT = 'ffac-selected-required-cpu-identity-v2'
EXTRA_MASK_NAMES = ('photo_sand_domain', 'feature_support', 'regional_native_domain',
    'flatfield_valid', 'feature_eligible', 'optical_presence_ge_half', 'frozen_gas', 'frozen_unresolved')


def require_frozen_gas_audit(audit):
    if (audit.get('contract') != 'ffac-selected-required-identity-code-consumer-audit-v1'
            or audit.get('complete') is not True
            or audit.get('frozen_gas_no_production_reads') is not True
            or audit.get('legacy_reconstruction_unreachable') is not True
            or audit.get('frozen_gas_compare_and_report_only') is not True):
        raise ValueError('No exact source audit permits diagnostic-only frozen_gas')


def inherited_identity(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.dtype != np.uint8 or expected.dtype != np.uint8 or actual.shape != expected.shape:
        raise ValueError('Inherited identity needs same-grid categorical uint8 maps')
    if not np.isin(actual, (0, 1, 2, 255)).all() or not np.isin(expected, (0, 1, 2, 255)).all():
        raise ValueError('Inherited map has a nonhistorical category')
    checks = {'co2_union': array_identity(np.isin(actual, (1, 2)), np.isin(expected, (1, 2))),
              'water': array_identity(actual == 0, expected == 0),
              'unresolved': array_identity(actual == 255, expected == 255)}
    return {'computed_from_scientific_images': True, 'archived_labels_used_as_inference_input': False,
            'comparisons': checks, 'all_bit_identical': all(row['bit_identical'] for row in checks.values())}


def compare_extra_masks(actual, expected, shape, *, frozen_gas_unreachable):
    actual, expected = np.asarray(actual), np.asarray(expected)
    count = int(np.prod(shape))
    if actual.shape != expected.shape or actual.shape != (8, (count + 7) // 8):
        raise ValueError('Extra-mask storage shape differs from original mask contract')
    rows = {}
    for index, name in enumerate(EXTRA_MASK_NAMES):
        check = array_identity(actual[index], expected[index])
        left = np.unpackbits(actual[index], bitorder='little')[:count].reshape(shape).astype(bool)
        right = np.unpackbits(expected[index], bitorder='little')[:count].reshape(shape).astype(bool)
        difference = left ^ right
        first = np.argwhere(difference)
        check.update(changed_pixels=int(difference.sum()),
                     first_yx=first[0].tolist() if len(first) else None,
                     added_pixels=int((left & ~right).sum()), removed_pixels=int((right & ~left).sum()),
                     required=not (name == 'frozen_gas' and frozen_gas_unreachable))
        rows[name] = check
    return {'comparisons': rows, 'frozen_gas_required': not frozen_gas_unreachable,
            'all_required_bit_identical': all(row['bit_identical'] for row in rows.values() if row['required']),
            'all_masks_bit_identical': all(row['bit_identical'] for row in rows.values())}


class StageConventionCentre:
    """Use optical inherited axes and G3 export axes from separate bindings."""
    def __init__(self, inner, inherited_conventions):
        self.inner = inner
        self.inherited_conventions = inherited_conventions

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def infer(self, *args, **kwargs):
        from . import phase_ffac_selected_native_centre as native
        from .tools import ffac_v5_temporal_alignment as temporal
        probability = kwargs.get('probability_receipt')
        if probability is None:
            raise ValueError('Required identity needs the matched original export receipt')
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


class RequiredCapturedCentre(CapturedCentre):
    def infer(self, *args, **kwargs):
        from . import phase_ffac_selected_native_centre as native
        original_co2 = native.original_co2_labels
        computed = {}

        def inherited(*positional, **named):
            labels, audit = original_co2(*positional, **named)
            computed['labels'] = labels.copy()
            return labels, audit

        with patch.object(native, 'original_co2_labels', inherited):
            result = super().infer(*args, **kwargs)
        if 'labels' not in computed:
            raise ValueError('No fresh inherited numerical map was computed')
        self.capture['inherited_recomputed'] = computed['labels']
        return result


class RequiredIdentityEvaluator(SelectedPhaseEvaluator):
    def __init__(self, root, plan, run):
        super().__init__(root, plan, run, backend='cpu')
        convention = plan['phase_campaign_binding']['inherited_stage_conventions'][run]
        for key in ('postflight_manifest', 'context'):
            bound_path(convention[key])
        audit_descriptor = plan['phase_campaign_binding']['frozen_gas_audit']
        audit = bound_json(audit_descriptor)
        require_frozen_gas_audit(audit)
        self.frozen_gas_audit = audit_descriptor
        self.centre = RequiredCapturedCentre(StageConventionCentre(self.centre, convention))

    def compare_saved_probabilities(self, source):
        from .phase_g3_probability_layers import _read_frozen_map
        descriptor = source['parent_source']['retained_source_cache']['parent_probability_receipt']
        expected = bound_json(descriptor)
        names = ('g3_pixel.npy', 'regional_frozen.npy', 'regional_refit.npy', 'co2.packbits.npy', 'extra_masks.packbits.npz')
        fields = ('g3_pixel', 'regional_frozen', 'regional_refit', 'co2', 'extra_masks')
        checks = {}
        masks = None
        for name, field in zip(names, fields):
            path = bound_path(expected['files'][name])
            if name.endswith('.npz'):
                with np.load(path, allow_pickle=False) as archive:
                    array = archive['masks']
                    masks = compare_extra_masks(self.centre.capture['stored'][field], array,
                        expected['metadata']['mask_shapes_hw'], frozen_gas_unreachable=True)
            else:
                array = np.load(path, allow_pickle=False)
            checks[name] = dict(array_identity(self.centre.capture['stored'][field], array), source=expected['files'][name])
        metadata = expected['metadata']
        native_descriptor = {'path': metadata['native_manifest_path'], 'sha256': metadata['native_manifest_sha256']}
        native = bound_json(native_descriptor)
        map_descriptor = {'path': metadata['frozen_map_path'], 'sha256': expected['binding']['frozen_map_sha256']}
        # The only historical labels read here are expected comparison data.
        # Fresh inherited inference has already completed and been captured.
        frozen, _ = _read_frozen_map(bound_path(map_descriptor), native['content_box_xywh'])
        inherited = inherited_identity(self.centre.capture['inherited_recomputed'], frozen)
        inherited.update(expected_original_stage_map=map_descriptor, expected_native_manifest=native_descriptor)
        first_four = all(checks[name]['bit_identical'] for name in names[:-1])
        return {'complete': True, 'receipt': descriptor, 'comparisons': checks,
            'extra_masks': masks, 'inherited_map': inherited, 'frozen_gas_audit': self.frozen_gas_audit,
            'all_stored_bit_identical_including_diagnostic': all(row['bit_identical'] for row in checks.values()),
            'all_bit_identical': first_four and masks['all_required_bit_identical'] and inherited['all_bit_identical']}


__all__ = ['RequiredIdentityEvaluator', 'StageConventionCentre', 'RequiredCapturedCentre',
           'inherited_identity', 'compare_extra_masks', 'require_frozen_gas_audit', 'CONTRACT']
