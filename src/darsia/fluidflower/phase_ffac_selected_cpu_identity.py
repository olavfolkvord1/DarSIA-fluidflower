"""CPU own-history identity instrumentation, separate from frozen producers.

The corrected numerical operator is called unchanged. Instrumentation records
its arrays and encodes the stored probability products for exact comparison.
It never accepts a tolerance, changes a history, reads annotations or fits.
"""
from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
import hashlib

import cv2
import numpy as np
from scipy.special import expit

from .phase_ffac_selected_inference_corrected import SelectedPhaseEvaluator
from .phase_ffac_selected_phase import asset, bound_json
from .phase_ffac_rig_camera_runtime import bound_path

CONTRACT = 'ffac-selected-cpu-own-history-identity-v1'


def array_identity(actual, expected):
    """Bitwise identity, including invalid values and their stored encodings."""
    first, second = np.asarray(actual), np.asarray(expected)
    same_grid = first.shape == second.shape and first.dtype == second.dtype
    if same_grid:
        byte_view = first.view(np.dtype((np.void, first.dtype.itemsize)))
        other_view = second.view(np.dtype((np.void, second.dtype.itemsize)))
        different = byte_view != other_view
        changed = int(np.count_nonzero(different))
        locations = np.argwhere(different)
        first_location = None if changed == 0 else locations[0].tolist()
    else:
        changed, first_location = None, None
    return dict(bit_identical=bool(same_grid and changed == 0), changed_elements=changed,
        first_different_index=first_location, actual_shape=list(first.shape),
        expected_shape=list(second.shape), actual_dtype=str(first.dtype), expected_dtype=str(second.dtype),
        actual_array_sha256=hashlib.sha256(np.ascontiguousarray(first).tobytes()).hexdigest(),
        expected_array_sha256=hashlib.sha256(np.ascontiguousarray(second).tobytes()).hexdigest())


def _stored_probabilities(g3, frozen, refit, flat_valid, labelled):
    if labelled:
        return dict(g3_pixel=g3.astype(np.float16),
            regional_frozen=cv2.resize(frozen,g3.shape[::-1],interpolation=cv2.INTER_LINEAR).astype(np.float16),
            regional_refit=cv2.resize(refit,g3.shape[::-1],interpolation=cv2.INTER_LINEAR).astype(np.float16))
    pixel=np.full(g3.shape,255,np.uint8)
    pixel[flat_valid]=np.rint(254*g3[flat_valid]).astype(np.uint8)
    return dict(g3_pixel=pixel,regional_frozen=frozen.astype(np.float16),regional_refit=refit.astype(np.float16))


class CapturedCentre:
    """Single-process instrumentation; wrapped numerical returns are unchanged."""
    def __init__(self, inner):
        self.inner=inner
        self.result=None
        self.capture={}

    def infer(self, *args, **kwargs):
        from . import phase_ffac_selected_native_centre as native
        from . import phase_gas_regional_context as gas_context
        from . import phase_optical_regions as regional
        from .tools.phase_inference_benchmark_backends import CpuFeaturePredictor
        original_co2=native.original_co2_labels
        original_encoding=native.encode_decode_probabilities
        original_shared=gas_context.build_shared_regional_context
        original_blocks=CpuFeaturePredictor.feature_blocks
        state={}

        def co2(*positional,**named):
            value=original_co2(*positional,**named)
            state.update(inherited_labels=value[0].copy(),current=positional[0],
                aligned_baseline=positional[1],generic_presence=positional[2],
                generic_gas=positional[3],domain=positional[4],spacing_m=np.asarray(positional[5]).tolist())
            return value

        def encoding(g3,reg,valid,**named):
            state.update(g3=g3,refit=reg,flat_valid=valid)
            return original_encoding(g3,reg,valid,**named)

        def shared(*positional,**named):
            value=original_shared(*positional,**named)
            state.update(shared=value,g3_presence=positional[2])
            return value

        def blocks(predictor,*positional,**named):
            old_gas=None
            for first,last,features in original_blocks(predictor,*positional,**named):
                if old_gas is None:
                    old_gas=np.empty(predictor.baseline['gray'].shape,np.float32)
                    state['old_gas_single_head']=old_gas
                old_gas[first:last]=expit(features @ predictor.weights[:,1]+predictor.bias[1]).reshape(last-first,old_gas.shape[1])
                yield first,last,features

        with ExitStack() as stack:
            stack.enter_context(patch.object(native,'original_co2_labels',co2))
            stack.enter_context(patch.object(native,'encode_decode_probabilities',encoding))
            stack.enter_context(patch.object(gas_context,'build_shared_regional_context',shared))
            stack.enter_context(patch.object(CpuFeaturePredictor,'feature_blocks',blocks))
            result=self.inner.infer(*args,**kwargs)
        # This unused legacy diagnostic layer was part of the historical
        # probability export. Computing it cannot change returned gas/CO2.
        legacy=np.where(state['flat_valid'],state['g3'],state['old_gas_single_head'])
        prepared=gas_context.prepare_conditional_gas(state['shared'],legacy)
        frozen=regional.probability(prepared['features'],self.inner.regional['heads']['gas_within_co2'])
        labelled=bool(kwargs.get('labelled_probability_encoding',False))
        stored=_stored_probabilities(state['g3'],frozen,state['refit'],state['flat_valid'],labelled)
        inherited=state['inherited_labels']
        masks=(np.asarray(kwargs['photo_sand_mask']),result.feature_support,result.domain,
            result.flatfield_valid,result.domain & result.flatfield_valid & (inherited!=255),
            state['g3_presence']>=.5,inherited==2,inherited==255)
        stored['co2']=np.packbits(np.isin(inherited,(1,2)).ravel(),bitorder='little')
        stored['extra_masks']=np.stack([np.packbits(mask.ravel(),bitorder='little') for mask in masks])
        self.result=result
        self.capture=dict(stored=stored,source_inputs=dict(spacing_m=state['spacing_m'],
            H_B_current=np.asarray(kwargs['H_B_current']).tolist(),
            H_B_previous=None if kwargs.get('H_B_previous') is None else np.asarray(kwargs['H_B_previous']).tolist(),
            baseline_origin_xy=list(kwargs['source_origin_xy']),previous_origin_xy=list(kwargs['previous_origin_xy']),
            current_valid_pixels=int(kwargs['current_valid'].sum()),
            baseline_valid_pixels=int(kwargs['baseline_valid'].sum()),
            previous_valid_pixels=None if kwargs['previous_valid'] is None else int(kwargs['previous_valid'].sum())),
            trace_arrays={name:dict(shape=list(np.asarray(value).shape),dtype=str(np.asarray(value).dtype),
                array_sha256=hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest())
                for name,value in [('decoded_current',state['current']),('aligned_baseline',state['aligned_baseline']),
                    ('generic_presence_two_head',state['generic_presence']),('generic_gas_two_head',state['generic_gas']),
                    ('g3_presence_single_head',state['g3_presence']),('feature_support',result.feature_support),
                    ('raw_feature_support',result.raw_feature_support),('native_domain',result.domain),
                    ('flatfield_valid',result.flatfield_valid),('original_co2_reconstruction',inherited)]})
        return result


class IdentityEvaluator(SelectedPhaseEvaluator):
    def __init__(self,root,plan,run):
        super().__init__(root,plan,run,backend='cpu')
        self.centre=CapturedCentre(self.centre)

    def compare_saved_probabilities(self,source):
        expected_descriptor=source['parent_source']['retained_source_cache']['parent_probability_receipt']
        expected=bound_json(expected_descriptor)
        actual=self.centre.capture['stored']
        names=('g3_pixel.npy','regional_frozen.npy','regional_refit.npy','co2.packbits.npy','extra_masks.packbits.npz')
        fields=('g3_pixel','regional_frozen','regional_refit','co2','extra_masks')
        comparisons={}
        for name,field in zip(names,fields):
            path=bound_path(expected['files'][name])
            if name.endswith('.npz'):
                with np.load(path,allow_pickle=False) as archive: value=archive['masks']
            else:value=np.load(path,allow_pickle=False)
            comparisons[name]=dict(array_identity(actual[field],value),source=expected['files'][name])
        return dict(complete=True,all_bit_identical=all(row['bit_identical'] for row in comparisons.values()),
            receipt=expected_descriptor,comparisons=comparisons)


__all__=['IdentityEvaluator','CapturedCentre','array_identity','CONTRACT']
